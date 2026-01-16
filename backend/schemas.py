from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ViolationBase(BaseModel):
    worker_id: str
    equipped_items: str
    violated_items: str
    evidence_path: str
    timestamp: Optional[datetime] = None
    worker_name: Optional[str] = None

class ViolationResponse(ViolationBase):
    id: int
    
    class Config:
        from_attributes = True

class WorkerBase(BaseModel):
    display_name: str
    
class WorkerResponse(WorkerBase):
    id: str
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class StatsResponse(BaseModel):
    total_workers: int
    helmet_count: int
    vest_count: int
    mask_count: int
    violations_today: int
