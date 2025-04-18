from abc import ABC, abstractmethod
from typing import List

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
    def clean(self, data_model: DataModel) -> List[DataModel]:
        pass


class NiceCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: NiceRawModel) -> List[NiceCleanedModel]:
        cleaned_models = []

        if data_model and data_model.chapters:
            for chapter in data_model.chapters:
                cleaned_text = ""
                if chapter and "markdown" in chapter and chapter["markdown"]:
                    cleaned_text = clean_text(chapter["markdown"])

                # Create a new NiceCleanedModel for each chapter
                chapter_model = NiceCleanedModel(
                    entry_id=data_model.entry_id,
                    title=data_model.title,
                    url=data_model.url,
                    last_updated=data_model.last_updated,
                    chapter=chapter["title"],
                    cleaned_content=cleaned_text,
                    type=data_model.type,
                )
                cleaned_models.append(chapter_model)

        # If no chapters, return an empty list
        return cleaned_models
