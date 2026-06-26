from fastapi import APIRouter, HTTPException
from sqlmodel import Session, select
from app.database import get_engine
from app.models.trace import Trace, Span

router = APIRouter(prefix="/api/v1/traces", tags=["traces"])


@router.get("")
def list_traces(limit: int = 50):
    with Session(get_engine()) as session:
        traces = list(session.exec(select(Trace).order_by(Trace.created_at.desc()).limit(limit)).all())
    return [
        {
            "trace_id": t.trace_id,
            "name": t.name,
            "status": t.status,
            "total_latency_ms": round(t.total_latency_ms, 1),
            "total_tokens": t.total_tokens,
            "created_at": t.created_at.isoformat(),
        }
        for t in traces
    ]


@router.get("/{trace_id}")
def get_trace(trace_id: str):
    with Session(get_engine()) as session:
        trace = session.exec(select(Trace).where(Trace.trace_id == trace_id)).first()
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        spans = list(session.exec(select(Span).where(Span.trace_id == trace_id).order_by(Span.created_at)).all())

    return {
        "trace_id": trace.trace_id,
        "name": trace.name,
        "status": trace.status,
        "total_latency_ms": round(trace.total_latency_ms, 1),
        "total_tokens": trace.total_tokens,
        "created_at": trace.created_at.isoformat(),
        "spans": [
            {
                "span_id": s.span_id,
                "parent_span_id": s.parent_span_id,
                "name": s.name,
                "provider": s.provider,
                "model": s.model,
                "status": s.status,
                "latency_ms": round(s.latency_ms, 1) if s.latency_ms else None,
                "tokens_in": s.tokens_in,
                "tokens_out": s.tokens_out,
                "input_text": s.input_text[:500] if s.input_text else None,
                "output_text": s.output_text[:500] if s.output_text else None,
                "error_message": s.error_message,
            }
            for s in spans
        ],
    }


@router.delete("/{trace_id}")
def delete_trace(trace_id: str):
    with Session(get_engine()) as session:
        spans = session.exec(select(Span).where(Span.trace_id == trace_id)).all()
        for s in spans:
            session.delete(s)
        trace = session.exec(select(Trace).where(Trace.trace_id == trace_id)).first()
        if trace:
            session.delete(trace)
        session.commit()
    return {"deleted": True}
