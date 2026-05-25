import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/admin", tags=["admin"])

REPORT_PATH = Path(os.environ.get("REPORT_JSON", "/photos/__data/status/report.json"))


@router.get("/report")
async def get_report():
    if not REPORT_PATH.exists():
        raise HTTPException(404, "report.json not found — run the Archive Report job first")
    import json
    return json.loads(REPORT_PATH.read_text())
