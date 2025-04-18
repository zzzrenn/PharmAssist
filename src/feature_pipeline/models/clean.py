import uuid
from typing import Tuple

from pydantic import Field

from models.base import VectorDBDataModel


class NiceCleanedModel(VectorDBDataModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entry_id: str
    title: str
    url: str
    last_updated: str
    chapter: str
    cleaned_content: str
    type: str

    def to_payload(self) -> Tuple[str, dict]:
        data = {
            "entry_id": self.entry_id,
            "title": self.title,
            "url": self.url,
            "last_updated": self.last_updated,
            "chapter": self.chapter,
            "cleaned_content": self.cleaned_content,
            "type": self.type,
        }

        return self.id, data
