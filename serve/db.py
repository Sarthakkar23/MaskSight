"""Inspection history: one row per /predict call.

SQLite by default; set DATABASE_URL (e.g. postgresql+psycopg://user:pass@host/db) to use PostgreSQL.
Images themselves live on disk under the storage directory; the database keeps their relative paths.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Inspection(Base):
    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    category: Mapped[str] = mapped_column(String(64), index=True)
    anomaly_score: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    is_defective: Mapped[bool]
    defect_area_fraction: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image_path: Mapped[str] = mapped_column(String(255), default="")
    mask_path: Mapped[str] = mapped_column(String(255), default="")

    def summary(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.replace(tzinfo=self.created_at.tzinfo or timezone.utc).isoformat(),
            "category": self.category,
            "anomaly_score": self.anomaly_score,
            "threshold": self.threshold,
            "is_defective": self.is_defective,
            "defect_area_fraction": self.defect_area_fraction,
            "latency_ms": self.latency_ms,
            "filename": self.filename,
        }


def make_engine(url: str) -> Engine:
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return engine


def make_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


def list_inspections(session: Session, limit: int, offset: int, category: str | None) -> list[Inspection]:
    q = select(Inspection).order_by(Inspection.id.desc()).limit(limit).offset(offset)
    if category:
        q = q.where(Inspection.category == category)
    return list(session.scalars(q))
