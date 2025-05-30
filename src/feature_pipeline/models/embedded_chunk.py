from typing import Tuple

import numpy as np

from models.base import VectorDBDataModel


class NiceEmbeddedChunkModel(VectorDBDataModel):
    entry_id: str
    chunk_id: str
    chunk_content: str
    embedded_content: list[float]
    title: str
    chapter: str
    url: str
    last_updated: str
    type: str

    class Config:
        arbitrary_types_allowed = True

    def to_payload(self) -> Tuple[str, np.ndarray, dict]:
        data = {
            "id": self.entry_id,
            "title": self.title,
            "chapter": self.chapter,
            "url": self.url,
            "last_updated": self.last_updated,
            "content": self.chunk_content,
            "type": self.type,
        }

        return self.chunk_id, self.embedded_content, data
