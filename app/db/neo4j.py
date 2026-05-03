from contextlib import asynccontextmanager
from neo4j import AsyncGraphDatabase
from app.config import settings

driver = None


def get_driver():
    return driver


async def init_driver():
    global driver
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )


async def close_driver():
    global driver
    if driver:
        await driver.close()


@asynccontextmanager
async def get_session():
    async with driver.session() as session:
        yield session
