import json
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from app.db.neo4j import get_session
from app.log import logger


def _ctx(request: Request) -> dict:
    return {
        "by": getattr(request.state, "user_email", None),
        "request_id": getattr(request.state, "request_id", None),
    }


router = APIRouter(prefix="/suggestions", tags=["suggestions"])

TYPE_LABELS = {
    'group_membership': 'Group Membership',
    'relationship':     'Family Relationship',
    'connection':       'Connection',
    'birth_year':       'Birth Year',
    'maiden_name':      'Maiden Name',
    'location':         'Group Location',
    'group_year':       'Group Year',
    'missing_ancestry': 'Missing Ancestry',
}


def _fmt(r):
    s = dict(r['s'])
    s['type_label'] = TYPE_LABELS.get(s.get('type'), s.get('type'))
    s['metadata']   = json.loads(s.get('metadata') or '{}')
    s['person']     = dict(r['p']) if r.get('p') else None
    if r.get('tp'):
        s['target'] = dict(r['tp']); s['target_kind'] = 'person'
    elif r.get('tg'):
        s['target'] = dict(r['tg']); s['target_kind'] = 'group'
    else:
        s['target'] = None; s['target_kind'] = None
    return s


@router.get("/")
async def list_suggestions(
    status:    str       = 'pending',
    type:      str | None = None,
    person_id: str | None = None,
):
    async with get_session() as session:
        params: dict = {'status': status}
        type_clause   = 'AND s.type = $type'      if type      else ''
        person_clause = 'AND (s.person_id = $person_id OR s.target_id = $person_id)' if person_id else ''
        if type:      params['type']      = type
        if person_id: params['person_id'] = person_id

        result = await session.run(
            f"""
            MATCH (s:Suggestion)
            WHERE s.status = $status {type_clause} {person_clause}
            OPTIONAL MATCH (p:Person  {{id: s.person_id}})
            OPTIONAL MATCH (tp:Person {{id: s.target_id}})
            OPTIONAL MATCH (tg:Group  {{id: s.target_id}})
            RETURN s, p, tp, tg
            ORDER BY s.confidence DESC, s.created_at DESC
            """,
            **params,
        )
        records = await result.data()
        return [_fmt(r) for r in records]


@router.get("/count")
async def count_suggestions():
    async with get_session() as session:
        result = await session.run(
            "MATCH (s:Suggestion {status: 'pending'}) RETURN count(s) AS n"
        )
        record = await result.single()
        return {'count': record['n'] if record else 0}


@router.post("/{suggestion_id}/accept", status_code=204)
async def accept_suggestion(suggestion_id: str, request: Request):
    async with get_session() as session:
        result = await session.run(
            "MATCH (s:Suggestion {id: $id}) SET s.status = 'accepted' RETURN s.id",
            id=suggestion_id,
        )
        if not await result.single():
            raise HTTPException(404, 'Not found')
    logger.bind(
        event="suggestion.accepted",
        suggestion_id=suggestion_id,
        **_ctx(request),
    ).info("suggestion accepted")


@router.post("/{suggestion_id}/reject", status_code=204)
async def reject_suggestion(suggestion_id: str, request: Request):
    async with get_session() as session:
        result = await session.run(
            "MATCH (s:Suggestion {id: $id}) SET s.status = 'rejected' RETURN s.id",
            id=suggestion_id,
        )
        if not await result.single():
            raise HTTPException(404, 'Not found')
    logger.bind(
        event="suggestion.rejected",
        suggestion_id=suggestion_id,
        **_ctx(request),
    ).info("suggestion rejected")


@router.post("/generate", status_code=202)
async def generate_suggestions(background_tasks: BackgroundTasks, request: Request):
    from app.suggestions.generator import run_all
    background_tasks.add_task(run_all)
    logger.bind(event="suggestions.generation_started", **_ctx(request)).info("suggestion generation started")
    return {'ok': True}
