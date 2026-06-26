import uuid
import time
from contextlib import asynccontextmanager
from sqlmodel import Session, select
from app.database import get_engine
from app.models.trace import Trace, Span


class TraceCollector:
    def __init__(self):
        self.engine = get_engine()

    def create_trace(self, name: str | None = None, metadata: dict | None = None) -> Trace:
        trace = Trace(
            trace_id=str(uuid.uuid4()),
            name=name,
            metadata_json=str(metadata) if metadata else None,
        )
        with Session(self.engine) as session:
            session.add(trace)
            session.commit()
            session.refresh(trace)
        return trace

    def add_span(
        self,
        trace_id: str,
        name: str,
        provider: str | None = None,
        model: str | None = None,
        parent_span_id: str | None = None,
    ) -> Span:
        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span_id,
            name=name,
            provider=provider,
            model=model,
        )
        with Session(self.engine) as session:
            session.add(span)
            session.commit()
            session.refresh(span)
        return span

    def finish_span(
        self,
        span_id: str,
        output: str | None = None,
        status: str = "success",
        latency_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        error_message: str | None = None,
    ):
        with Session(self.engine) as session:
            span = session.get(Span, span_id)
            if span:
                span.output_text = output
                span.status = status
                span.latency_ms = latency_ms
                span.tokens_in = tokens_in
                span.tokens_out = tokens_out
                span.error_message = error_message
                session.add(span)
                session.commit()

    def finish_trace(self, trace_id: str, status: str = "success"):
        with Session(self.engine) as session:
            trace = session.exec(select(Trace).where(Trace.trace_id == trace_id)).first()
            if trace:
                spans = session.exec(select(Span).where(Span.trace_id == trace_id)).all()
                trace.total_latency_ms = sum(s.latency_ms or 0 for s in spans)
                trace.total_tokens = sum((s.tokens_in or 0) + (s.tokens_out or 0) for s in spans)
                trace.status = status
                session.add(trace)
                session.commit()

    @asynccontextmanager
    async def trace_call(self, name: str, provider: str, model: str):
        trace = self.create_trace(name=name)
        span = self.add_span(trace_id=trace.trace_id, name=name, provider=provider, model=model)

        start = time.perf_counter()
        status = "success"
        output = None
        error = None
        tokens_in = 0
        tokens_out = 0

        try:
            yield trace, span
        except Exception as e:
            status = "error"
            error = str(e)
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            self.finish_span(
                span_id=span.span_id,
                output=output,
                status=status,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                error_message=error,
            )
            self.finish_trace(trace_id=trace.trace_id, status=status)
