from fastapi import APIRouter
from sqlmodel import Session, select, func
from app.database import get_engine
from app.models.eval import EvalRun, EvalResult
from app.models.trace import Trace, ProviderUsage

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/summary")
def get_summary():
    with Session(get_engine()) as session:
        total_runs = session.exec(select(func.count(EvalRun.id))).one()
        total_traces = session.exec(select(func.count(Trace.id))).one()
        total_results = session.exec(select(func.count(EvalResult.id))).one()
        passed = session.exec(select(func.count(EvalResult.id)).where(EvalResult.status == "pass")).one()
        failed = session.exec(select(func.count(EvalResult.id)).where(EvalResult.status == "fail")).one()

    return {
        "total_eval_runs": total_runs,
        "total_traces": total_traces,
        "total_eval_results": total_results,
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{(passed / total_results * 100):.1f}%" if total_results > 0 else "N/A",
    }


@router.get("/providers")
def get_provider_stats():
    with Session(get_engine()) as session:
        runs = list(session.exec(select(EvalRun)).all())

    stats = {}
    for r in runs:
        key = f"{r.provider}/{r.model}"
        if key not in stats:
            stats[key] = {"calls": 0, "total_tokens": 0, "total_cost": 0.0, "total_latency": 0.0}
        stats[key]["calls"] += 1
        stats[key]["total_tokens"] += r.total_tokens
        stats[key]["total_cost"] += r.estimated_cost
        stats[key]["total_latency"] += r.avg_latency_ms * r.total_cases

    return [
        {
            "provider_model": k,
            "calls": v["calls"],
            "total_tokens": v["total_tokens"],
            "total_cost": round(v["total_cost"], 4),
            "avg_latency_ms": round(v["total_latency"] / v["calls"], 1) if v["calls"] > 0 else 0,
        }
        for k, v in stats.items()
    ]


@router.get("/failures")
def get_failure_summary():
    with Session(get_engine()) as session:
        results = list(
            session.exec(select(EvalResult).where(EvalResult.status.in_(["fail", "error"]))).all()
        )

    failure_types = {}
    for r in results:
        key = r.error_message if r.error_message else "wrong_output"
        if key not in failure_types:
            failure_types[key] = 0
        failure_types[key] += 1

    return [{"type": k, "count": v} for k, v in sorted(failure_types.items(), key=lambda x: -x[1])[:10]]
