from typing import List
from models.base import DataModel


class NiceRawModel(DataModel):
    title: str
    url: str
    last_updated: str
    chapters: List[dict]