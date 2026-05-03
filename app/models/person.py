from pydantic import BaseModel
from typing import Optional


class Person(BaseModel):
    id: str
    name: str
    known_as: Optional[str] = None
    maiden_name: Optional[str] = None
    former_names: Optional[list[str]] = None
    gender: Optional[str] = None
    birth_date: Optional[str] = None
    birth_date_precision: Optional[str] = None
    birth_place: Optional[str] = None
    death_date: Optional[str] = None
    death_date_precision: Optional[str] = None
    death_place: Optional[str] = None
    is_living: bool = True
    notes: Optional[str] = None
