from fastapi import FastAPI, UploadFile, File, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil
import os
import cv2 as cv
from pathlib import Path
from typing import List

from backend.schemas import ViolationResponse, StatsResponse, WorkerResponse
from backend.pipeline_service import PPEPipeline
from database.database import get_recent_violations, get_all_workers

app = FastAPI(title="PPE Detection System")

from fastapi.staticfiles import StaticFiles

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "videos", "uploads")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

# Mount Storage for Static Access (Snapshots)
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")

# Global Pipeline Instance (Singleton-ish for now)
pipeline_instance = PPEPipeline()

@app.get("/")
def read_root():
    return {"status": "System Operational"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Initialize pipeline with new video
    pipeline_instance.set_source(file_location)
    
    return {"filename": file.filename, "status": "Uploaded and Pipeline Initialized"}

@app.get("/video_feed")
def video_feed():
    """
    Stream video frames from the pipeline.
    """
    return StreamingResponse(pipeline_instance.generate_frames(), 
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/violations", response_model=List[ViolationResponse])
def get_violations():
    # Fetch violations from current session
    violations = get_recent_violations(limit=50, from_timestamp=pipeline_instance.session_start_time)
    
    return [ViolationResponse(
        id=v['id'],
        worker_id=str(v['worker_id']), 
        equipped_items=v['equipped_items'], 
        violated_items=v['violated_items'],
        evidence_path=os.path.basename(v['evidence_path']) if v['evidence_path'] else "", # Return only filename
        timestamp=v['timestamp'],
        worker_name=v.get('worker_name')
    ) for v in violations]

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    # TODO: Aggregate from pipeline or DB
    return pipeline_instance.get_stats()

@app.get("/workers", response_model=List[WorkerResponse])
def get_workers():
    workers = get_all_workers()
    return [WorkerResponse(
        id=w['id'],
        display_name=w['display_name'],
        created_at=w['created_at']
    ) for w in workers]
