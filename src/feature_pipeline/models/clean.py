from typing import Tuple

from models.base import VectorDBDataModel


class NiceCleanedModel(VectorDBDataModel):
    entry_id: str
    title: str
    url: str
    last_updated: str
    cleaned_content: str
    type: str

    def to_payload(self) -> Tuple[str, dict]:
        data = {
            "title": self.title,
            "url": self.url,
            "last_updated": self.last_updated,
            "cleaned_content": self.cleaned_content,
            "type": self.type,
        }

        return self.entry_id, data
