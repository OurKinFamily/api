from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # Root of the photos volume — change this if the drive moves
    photos_root: Path = Path("/photos")

    class Config:
        env_file = ".env"


settings = Settings()
