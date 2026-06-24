from sqlmodel import SQLModel, create_engine, Session
from app.config import get_settings
import os

engine = None


def get_engine():
    global engine
    if engine is None:
        settings = get_settings()
        db_path = settings.database_url.replace("sqlite+aiosqlite:///", "")
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        engine = create_engine(settings.database_url.replace("sqlite+aiosqlite", "sqlite"))
    return engine


def init_db():
    eng = get_engine()
    SQLModel.metadata.create_all(eng)


def get_session():
    eng = get_engine()
    with Session(eng) as session:
        yield session
