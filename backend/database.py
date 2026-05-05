from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DocumentResult(Base):
    __tablename__ = "document_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    document_type = Column(String, nullable=True)
    quantum_analysis = Column(String, nullable=True)
    name = Column(String, nullable=True)
    date = Column(String, nullable=True)
    amount = Column(String, nullable=True)
    skills = Column(String, nullable=True)
    summary = Column(String, nullable=True)
    key_phrases = Column(String, nullable=True)
    full_text = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
