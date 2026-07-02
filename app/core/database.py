import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# base class that all our table models inherit from
Base = declarative_base()


def get_db():
    """
    FastAPI dependency - gives each request a DB session, closes it after.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()