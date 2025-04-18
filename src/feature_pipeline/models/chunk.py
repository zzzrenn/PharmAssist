from models.base import DataModel


class NiceChunkModel(DataModel):
    id: str
    entry_id: str
    chunk_id: str
    chunk_content: str
    title: str
    chapter: str
    url: str
    last_updated: str
    type: str
