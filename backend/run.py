import uvicorn
from app.config import settings

if __name__ == "__main__":
    # Start FastAPI server on port 8000
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
