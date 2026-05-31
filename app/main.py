from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db.neo4j import init_driver, close_driver
from app.log import logger
from app.middleware.auth import AuthMiddleware
from app.middleware.request_log import RequestLogMiddleware
from app.routers import people, media, jobs, gallery, faces, admin, places, groups, suggestions, heritage, me, albums, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("api.starting")
    await init_driver()
    logger.info("api.ready")
    yield
    await close_driver()
    logger.info("api.stopped")


app = FastAPI(title="ourkin API", lifespan=lifespan)

# Order matters: RequestLog wraps AuthMiddleware so we get a log line even
# when auth rejects. AuthMiddleware runs first chronologically because
# Starlette walks middleware in reverse-add order.
app.add_middleware(AuthMiddleware)
app.add_middleware(RequestLogMiddleware)

app.include_router(people.router)
app.include_router(media.router)
app.include_router(jobs.router)
app.include_router(gallery.router)
app.include_router(faces.router)
app.include_router(admin.router)
app.include_router(places.router)
app.include_router(groups.router)
app.include_router(suggestions.router)
app.include_router(heritage.router)
app.include_router(me.router)
app.include_router(albums.router)
app.include_router(search.router)
