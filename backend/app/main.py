import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import engine, Base, ensure_hazard_schema
from app.api import endpoints
from app.routes import hazards
from app.services.video_processor import video_processor

# Create SQLite tables on startup (simple for MVP)
Base.metadata.create_all(bind=engine)
ensure_hazard_schema()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.on_event("startup")
def startup_event():
    video_processor.start()

@app.on_event("shutdown")
def shutdown_event():
    video_processor.stop()

# Set up CORS middleware for local React dev server
origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]
if "*" in origins:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the endpoints router
app.include_router(endpoints.router, prefix=settings.API_V1_STR)
app.include_router(hazards.router, prefix=settings.API_V1_STR)

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

@app.get("/")
def read_root():
    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "api_docs": "/docs"
    }
