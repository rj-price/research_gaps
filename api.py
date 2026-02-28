import os
import uuid
import asyncio
import shutil
import logging
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from aiolimiter import AsyncLimiter

from modules.llm import get_client, summarise_paper, identify_gaps
from modules.db import init_db

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Research Gap Identifier API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task store (Use Redis/Database for production)
tasks: Dict[str, Dict] = {}

class AnalysisRequest(BaseModel):
    subject: str = "the provided topics"
    model: str = "gemini-2.5-flash"
    rate_limit: int = 5
    concurrent_requests: int = 5

@app.on_event("startup")
async def startup_event():
    await init_db()
    # Ensure temp directory exists
    os.makedirs("temp_uploads", exist_ok=True)

async def run_analysis(task_id: str, subject: str, model: str, rate_limit: int, concurrent_requests: int, file_paths: List[str]):
    tasks[task_id]["status"] = "processing"
    try:
        client = get_client()
        limiter = AsyncLimiter(rate_limit, 60)
        semaphore = asyncio.Semaphore(concurrent_requests)

        async def bounded_summarise(pdf_path: str) -> str:
            async with semaphore:
                return await summarise_paper(client, model, pdf_path, limiter)

        # Process all PDFs
        process_tasks = [bounded_summarise(pdf) for pdf in file_paths]
        
        paper_summaries = []
        for f in asyncio.as_completed(process_tasks):
            summary = await f
            paper_summaries.append(summary)

        valid_summaries = [s for s in paper_summaries if not s.startswith("Error summarising")]

        if not valid_summaries:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = "No valid summaries generated."
            return

        logger.info(f"Task {task_id}: Analysing research gaps...")
        report = await identify_gaps(client, model, valid_summaries, subject, limiter)

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = {
            "report": report,
            "summaries": valid_summaries
        }
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
    finally:
        # Cleanup temp files
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)

@app.post("/analyse")
async def analyse(
    background_tasks: BackgroundTasks,
    subject: str = "the provided topics",
    model: str = "gemini-2.5-flash",
    rate_limit: int = 5,
    concurrent_requests: int = 5,
    files: List[UploadFile] = File(...)
):
    task_id = str(uuid.uuid4())
    file_paths = []
    
    for file in files:
        if not file.filename.endswith(".pdf"):
            continue
        
        file_path = f"temp_uploads/{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)

    if not file_paths:
        raise HTTPException(status_code=400, detail="No valid PDF files uploaded.")

    tasks[task_id] = {"status": "pending", "subject": subject}
    background_tasks.add_task(run_analysis, task_id, subject, model, rate_limit, concurrent_requests, file_paths)
    
    return {"task_id": task_id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
