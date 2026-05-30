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


# Browser cache-bust token for /api/media/* URLs. Bump in lockstep with the
# frontend constant in app/src/lib/media.js after regenerating thumbs/medium.
MEDIA_VERSION = 1


def with_v(url: str | None) -> str | None:
    """Append ?v=MEDIA_VERSION (or &v=) to a /api/media/* URL for cache busting."""
    if not url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}v={MEDIA_VERSION}"
