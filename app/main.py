from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db.neo4j import init_driver, close_driver
from app.middleware.auth import AuthMiddleware
from app.routers import people, media


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_driver()
    yield
    await close_driver()


app = FastAPI(title="ourkin API", lifespan=lifespan)

app.add_middleware(AuthMiddleware)

app.include_router(people.router)
app.include_router(media.router)
