from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.eval_runner import EvalRunner

router = APIRouter(prefix="/api/v1/eval", tags=["eval"])
runner = EvalRunner()


class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None


class TestCaseCreate(BaseModel):
    input: str
    expected: str
    category: Optional[str] = None
    difficulty: str = "medium"


class DatasetImport(BaseModel):
    cases: list[TestCaseCreate]


class RunCreate(BaseModel):
    dataset_id: int
    provider: str
    model: str
    prompt_template: Optional[str] = None


@router.post("/datasets")
def create_dataset(body: DatasetCreate):
    dataset = runner.create_dataset(name=body.name, description=body.description)
    return {"id": dataset.id, "name": dataset.name}


@router.get("/datasets")
def list_datasets():
    datasets = runner.list_datasets()
    return [{"id": d.id, "name": d.name, "description": d.description, "created_at": d.created_at.isoformat()} for d in datasets]


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: int):
    dataset = runner.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    cases = runner.get_test_cases(dataset_id)
    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "cases": [{"id": c.id, "input": c.input_text, "expected": c.expected_output, "category": c.category, "difficulty": c.difficulty} for c in cases],
    }


@router.post("/datasets/{dataset_id}/cases")
def add_test_cases(dataset_id: int, body: DatasetImport):
    dataset = runner.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    cases = [c.model_dump() for c in body.cases]
    count = runner.add_test_cases(dataset_id, cases)
    return {"added": count}


@router.post("/runs")
async def create_and_run_eval(body: RunCreate):
    run = runner.create_run(
        dataset_id=body.dataset_id,
        provider_name=body.provider,
        model=body.model,
        prompt_template=body.prompt_template,
    )
    run = await runner.execute_run(run.id)
    return {
        "id": run.id,
        "status": run.status,
        "total_cases": run.total_cases,
        "passed": run.passed_cases,
        "failed": run.failed_cases,
        "pass_rate": f"{(run.passed_cases / run.total_cases * 100):.1f}%" if run.total_cases > 0 else "N/A",
        "avg_latency_ms": round(run.avg_latency_ms, 1),
        "total_tokens": run.total_tokens,
        "estimated_cost": round(run.estimated_cost, 4),
    }


@router.get("/runs")
def list_runs(dataset_id: Optional[int] = None):
    runs = runner.list_runs(dataset_id=dataset_id)
    return [
        {
            "id": r.id,
            "dataset_id": r.dataset_id,
            "provider": r.provider,
            "model": r.model,
            "status": r.status,
            "total_cases": r.total_cases,
            "passed": r.passed_cases,
            "failed": r.failed_cases,
            "avg_latency_ms": round(r.avg_latency_ms, 1),
            "estimated_cost": round(r.estimated_cost, 4),
            "created_at": r.created_at.isoformat(),
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        }
        for r in runs
    ]


@router.get("/runs/{run_id}")
def get_run(run_id: int):
    run = runner.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    results = runner.get_run_results(run_id)
    return {
        "id": run.id,
        "provider": run.provider,
        "model": run.model,
        "status": run.status,
        "total_cases": run.total_cases,
        "passed": run.passed_cases,
        "failed": run.failed_cases,
        "avg_latency_ms": round(run.avg_latency_ms, 1),
        "total_tokens": run.total_tokens,
        "estimated_cost": round(run.estimated_cost, 4),
        "results": [
            {
                "id": r.id,
                "test_case_id": r.test_case_id,
                "status": r.status,
                "actual_output": r.actual_output,
                "score": r.score,
                "latency_ms": r.latency_ms,
                "error_message": r.error_message,
            }
            for r in results
        ],
    }
