"""LLM-powered unified search.

POST /search  body: { "q": str }
Returns: { media: [...], debug: { ... } }

Debug payload is rich so the SearchPage can show what's happening end-to-end:
the system prompt sent, raw LLM response, parsed plan, resolved entities,
final Cypher + bind params, timings.
"""

import time
from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.db.neo4j import get_session
from app.log import logger as log
from app.services import search as svc


router = APIRouter(prefix="/search", tags=["search"])


class SearchTurn(BaseModel):
    q: str
    plan: dict | None = None


class SearchBody(BaseModel):
    q: str = Field(..., min_length=1, max_length=512)
    # Optional prior turns — server passes to LLM as conversational context.
    # Frontend should trim to recent turns; server hard-caps at 3 anyway.
    history: list[SearchTurn] = Field(default_factory=list)


@router.post("")
async def search(body: SearchBody, request: Request):
    q = body.q.strip()
    t0 = time.time()
    today = datetime.now().strftime("%Y-%m-%d")
    history = [t.model_dump() for t in body.history][-3:]

    plan_result = await svc.plan_query(q, today, history=history)
    plan        = plan_result.get("plan") or {}
    t_plan_ms   = plan_result.get("ms", 0)

    media: list = []
    cypher = ""
    params: dict = {}
    resolved: dict = {}
    dropped: list[str] = []
    total_count: int | None = None
    fact_result: dict | None = None
    intent = (plan.get("intent") or "media_search").lower()
    if intent not in ("media_search", "count", "fact", "unknown"):
        intent = "media_search"

    async with get_session() as session:
        resolved = await svc.resolve_plan(plan, session)
        t_resolve_ms = int((time.time() - t0) * 1000) - t_plan_ms

        has_signal = bool(
            resolved.get("person_ids")
            or resolved.get("place")
            or resolved.get("date_match_resolved")
            or plan.get("year_from") or plan.get("year_to")
            or plan.get("media_type") and plan.get("media_type") != "all"
        )
        t_exec_start = time.time()
        try:
            if intent == "fact":
                fact_result = await svc.fact_plan(plan, resolved, session)
            elif intent == "count" and has_signal:
                total_count, cypher, params = await svc.count_plan(plan, resolved, session)
                # Also fetch a small sample so the UI can show thumbnails
                sample_plan = {**plan, "limit": 12}
                media, _, _, _ = await svc.execute_plan(sample_plan, resolved, session)
            elif intent == "media_search" and has_signal:
                media, cypher, params, dropped = await svc.execute_plan(plan, resolved, session)
        except Exception as e:
            log.warning(f"search dispatch failed: {e}")
            cypher = f"-- ERROR: {e}"
        t_exec_ms = int((time.time() - t_exec_start) * 1000)

    total_ms = int((time.time() - t0) * 1000)

    log.bind(
        event="search.executed",
        q=q,
        intent=intent,
        result_count=len(media),
        total_count=total_count,
        plan=plan,
        person_ids=resolved.get("person_ids"),
        place_slug=(resolved.get("place") or {}).get("slug"),
        total_ms=total_ms,
        by=getattr(request.state, "user_email", None),
        request_id=getattr(request.state, "request_id", None),
    ).info("search executed")

    answer_card = svc.build_answer_card(
        q, plan, resolved, dropped, len(media),
        intent=intent, total_count=total_count, fact_result=fact_result,
    )

    return {
        "intent":      intent,
        "media":       media,
        "total_count": total_count,
        "fact":        fact_result,
        "dropped":     dropped,
        "error":       plan_result.get("error"),
        "answer":      answer_card,
        "debug": {
            "query":          q,
            "intent":         intent,
            "error":          plan_result.get("error"),
            "model":          plan_result.get("model"),
            "cached":         plan_result.get("cached"),
            "system_prompt":  plan_result.get("system_prompt"),
            "raw_response":   plan_result.get("raw_response"),
            "plan":           plan,
            "resolved":       resolved,
            "cypher":         cypher,
            "cypher_params":  params,
            "total_count":    total_count,
            "fact":           fact_result,
            "timings_ms": {
                "plan":    t_plan_ms,
                "resolve": t_resolve_ms,
                "execute": t_exec_ms,
                "total":   total_ms,
            },
            "has_structured_signal": has_signal,
        },
    }
