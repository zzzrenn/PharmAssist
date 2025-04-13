from models.base import DataModel


class NiceChunkModel(DataModel):
    entry_id: str
    chunk_id: str
    chunk_content: str
    title: str
    url: str
    last_updated: str
    type: str
