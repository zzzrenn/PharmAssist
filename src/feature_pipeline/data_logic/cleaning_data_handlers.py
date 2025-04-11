from abc import ABC, abstractmethod

from models.base import DataModel
from models.clean import NiceCleanedModel
from models.raw import NiceRawModel
from utils.cleaning import clean_text


class CleaningDataHandler(ABC):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """

    @abstractmethod
    def clean(self, data_model: DataModel) -> DataModel:
        pass


class NiceCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: NiceRawModel) -> NiceCleanedModel:
        # Join all chapter markdown content with newlines between chapters
        joined_text = "\n\n".join(
            chapter["markdown"] for chapter in data_model.chapters
        ) if data_model and data_model.chapters else None

        return NiceCleanedModel(
            entry_id=data_model.entry_id,
            title=data_model.title,
            url=data_model.url,
            last_updated=data_model.last_updated,
            cleaned_content=clean_text(joined_text),
            type=data_model.type,
        )