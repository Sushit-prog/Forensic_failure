from sqlmodel import Session, select
from datetime import datetime
from app.database import get_engine
from app.models.eval import EvalDataset, TestCase, EvalRun, EvalResult
from app.services.providers.registry import get_provider
from app.services.scoring import exact_match_score, semantic_similarity_score
from app.services.trace_collector import TraceCollector


class EvalRunner:
    def __init__(self):
        self.engine = get_engine()
        self.tracer = TraceCollector()

    def create_dataset(self, name: str, description: str | None = None) -> EvalDataset:
        dataset = EvalDataset(name=name, description=description)
        with Session(self.engine) as session:
            session.add(dataset)
            session.commit()
            session.refresh(dataset)
        return dataset

    def add_test_cases(self, dataset_id: int, cases: list[dict]) -> int:
        count = 0
        with Session(self.engine) as session:
            for case in cases:
                tc = TestCase(
                    dataset_id=dataset_id,
                    input_text=case["input"],
                    expected_output=case["expected"],
                    category=case.get("category"),
                    difficulty=case.get("difficulty", "medium"),
                )
                session.add(tc)
                count += 1
            session.commit()
        return count

    def get_dataset(self, dataset_id: int) -> EvalDataset | None:
        with Session(self.engine) as session:
            return session.get(EvalDataset, dataset_id)

    def get_test_cases(self, dataset_id: int) -> list[TestCase]:
        with Session(self.engine) as session:
            return list(session.exec(select(TestCase).where(TestCase.dataset_id == dataset_id)).all())

    def list_datasets(self) -> list[EvalDataset]:
        with Session(self.engine) as session:
            return list(session.exec(select(EvalDataset)).all())

    def create_run(self, dataset_id: int, provider_name: str, model: str, prompt_template: str | None = None) -> EvalRun:
        run = EvalRun(
            dataset_id=dataset_id,
            provider=provider_name,
            model=model,
            prompt_template=prompt_template,
        )
        with Session(self.engine) as session:
            session.add(run)
            session.commit()
            session.refresh(run)
        return run

    async def execute_run(self, run_id: int) -> EvalRun:
        provider = None
        with Session(self.engine) as session:
            run = session.get(EvalRun, run_id)
            if not run:
                raise ValueError(f"Run {run_id} not found")
            cases = list(session.exec(select(TestCase).where(TestCase.dataset_id == run.dataset_id)).all())
            provider = get_provider(run.provider)

        total_latency = 0.0
        total_tokens = 0
        total_cost = 0.0
        passed = 0
        failed = 0

        for case in cases:
            try:
                async with self.tracer.trace_call(
                    name=f"eval_case_{case.id}",
                    provider=run.provider,
                    model=run.model,
                ) as (trace, span):
                    response = await provider.complete(
                        prompt=case.input_text,
                        model=run.model,
                        system_prompt=run.prompt_template,
                    )

                    score = exact_match_score(case.expected_output, response.content)
                    if score < 1.0:
                        sem_score = semantic_similarity_score(case.expected_output, response.content)
                        score = max(score, sem_score)

                    status = "pass" if score >= 0.8 else "fail"

                    result = EvalResult(
                        run_id=run_id,
                        test_case_id=case.id,
                        status=status,
                        actual_output=response.content,
                        score=score,
                        latency_ms=response.latency_ms,
                        tokens_in=response.tokens_in,
                        tokens_out=response.tokens_out,
                    )
                    with Session(self.engine) as session:
                        session.add(result)
                        session.commit()

                    total_latency += response.latency_ms
                    total_tokens += response.tokens_in + response.tokens_out
                    total_cost += provider.estimate_cost(response.tokens_in, response.tokens_out)

                    if status == "pass":
                        passed += 1
                    else:
                        failed += 1

            except Exception as e:
                result = EvalResult(
                    run_id=run_id,
                    test_case_id=case.id,
                    status="error",
                    error_message=str(e),
                )
                with Session(self.engine) as session:
                    session.add(result)
                    session.commit()
                failed += 1

        total_cases = passed + failed
        with Session(self.engine) as session:
            run = session.get(EvalRun, run_id)
            if run:
                run.total_cases = total_cases
                run.passed_cases = passed
                run.failed_cases = failed
                run.avg_latency_ms = total_latency / total_cases if total_cases > 0 else 0
                run.total_tokens = total_tokens
                run.estimated_cost = total_cost
                run.status = "completed"
                run.completed_at = datetime.utcnow()
                session.add(run)
                session.commit()
                session.refresh(run)
                return run

        return run

    def get_run(self, run_id: int) -> EvalRun | None:
        with Session(self.engine) as session:
            return session.get(EvalRun, run_id)

    def get_run_results(self, run_id: int) -> list[EvalResult]:
        with Session(self.engine) as session:
            return list(session.exec(select(EvalResult).where(EvalResult.run_id == run_id)).all())

    def list_runs(self, dataset_id: int | None = None) -> list[EvalRun]:
        with Session(self.engine) as session:
            if dataset_id:
                return list(session.exec(select(EvalRun).where(EvalRun.dataset_id == dataset_id)).all())
            return list(session.exec(select(EvalRun)).all())
