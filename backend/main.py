"""
Main entry point for the Resume Analysis API
This file imports and uses the main API from api.py
"""

from api import app
from database import init_database

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on application startup"""
    print("🚀 Starting Resume Analysis API...")
    init_database()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
