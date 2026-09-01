from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

# SQLite configuration details
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL, connect_args=connect_args
)

# Enforce SQLite foreign keys constraint (mandatory for SQLite relationships)
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_hazard_schema() -> None:
    """Add required columns when opening an SQLite database created by an older MVP build."""
    if not settings.DATABASE_URL.startswith("sqlite"):
        return
    with engine.begin() as connection:
        columns = {column["name"] for column in inspect(connection).get_columns("hazards")} if inspect(connection).has_table("hazards") else set()
        if "timestamp" not in columns:
            connection.execute(text("ALTER TABLE hazards ADD COLUMN timestamp DATETIME"))
            connection.execute(text("UPDATE hazards SET timestamp = COALESCE(first_detected, CURRENT_TIMESTAMP)"))
        if "source" not in columns:
            connection.execute(text("ALTER TABLE hazards ADD COLUMN source VARCHAR(100) NOT NULL DEFAULT 'manual'"))
        if "is_simulated" not in columns:
            connection.execute(text("ALTER TABLE hazards ADD COLUMN is_simulated INTEGER NOT NULL DEFAULT 0"))
        if "gps_source" not in columns:
            connection.execute(text("ALTER TABLE hazards ADD COLUMN gps_source VARCHAR(100) NOT NULL DEFAULT 'unknown'"))
        if "gps_is_simulated" not in columns:
            connection.execute(text("ALTER TABLE hazards ADD COLUMN gps_is_simulated INTEGER NOT NULL DEFAULT 0"))
        observation_columns = {column["name"] for column in inspect(connection).get_columns("observations")} if inspect(connection).has_table("observations") else set()
        if observation_columns and "is_simulated" not in observation_columns:
            connection.execute(text("ALTER TABLE observations ADD COLUMN is_simulated INTEGER NOT NULL DEFAULT 0"))
        if observation_columns and "gps_source" not in observation_columns:
            connection.execute(text("ALTER TABLE observations ADD COLUMN gps_source VARCHAR(100) NOT NULL DEFAULT 'unknown'"))
        if observation_columns and "gps_is_simulated" not in observation_columns:
            connection.execute(text("ALTER TABLE observations ADD COLUMN gps_is_simulated INTEGER NOT NULL DEFAULT 0"))
