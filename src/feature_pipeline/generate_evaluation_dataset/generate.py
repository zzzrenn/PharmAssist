import os
import sys

# Add the project root to path to resolve module imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import json
from pathlib import Path
from typing import Any, Dict, List

from comet_ml import Artifact, start
from openai import OpenAI
from qdrant_client import models

from core import get_logger
from core.config import settings
from core.db.qdrant import QdrantDatabaseConnector

logger = get_logger(__name__)

client = QdrantDatabaseConnector()


class EvaluationQuestionGenerator:
    def __init__(
        self,
        openai_model: str = settings.OPENAI_MODEL_ID,
        n_questions_per_content: int = 3,
    ):
        """
        Initialize the evaluation question generator.

        Args:
            openai_model: OpenAI model to use for question generation
        """
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.openai_model = openai_model
        self.max_length = 16384
        self.system_prompt = self.get_system_prompt()
        self.n_questions_per_content = n_questions_per_content

    def get_system_prompt(self) -> str:
        """Get the system prompt for question generation."""
        return """You are an expert medical/pharmaceutical question generator and answer provider, specialized in creating comprehensive and clinically relevant evaluation questions and their corresponding gold-standard answers directly from medical guideline text. Your goal is to generate questions that holistically cover the most important and actionable parts of the provided document, along with accurate answers strictly derived from the same content.

        For each piece of content, generate distinct and challenging questions of different types. **For each question, also provide a concise and accurate ground truth answer that is strictly derived from the provided content.**

        1.  **FACTUAL:** Direct questions about specific facts, definitions, procedures, or quantitative data. These should assess basic recall and understanding.
        2.  **COMPARATIVE:** Questions requiring the comparison or contrast of treatments, conditions, diagnoses, outcomes, or patient characteristics. These should highlight differences, similarities, and trade-offs.
        3.  **CLINICAL APPLICATION/INTERPRETATIVE:** Questions requiring analysis, reasoning, synthesis of information, or the application of guidelines to a hypothetical clinical scenario. These should probe deeper understanding and critical judgment, asking "why" or "how to apply."

        Structure your response as a JSON list, ready for `json.loads()`, with each object containing:
        -   `"query"`: The generated question.
        -   `"query_type"`: One of "factual", "comparative", or "clinical_application".
        -   `"gt_answer"`: The concise and accurate answer to the question, **strictly derived from the provided content only**.

        Ensure questions are:
        -   **Strictly answerable using *only* the information explicitly provided in the source document.** This is paramount to ensure the questions are grounded.
        -   **Highly Relevant:** Focus on the most critical information, key recommendations, risks, benefits, and decision points within the medical guideline.
        -   **Holistic and Broad:** Aim to cover different sections or aspects of the provided content, not just one isolated point.
        -   Clinically relevant and practical.
        -   Clear, unambiguous, and concise.
        -   Varied in complexity, moving from basic recall to complex application.

        Ensure `ground_truth_answer` is:
        -   **Completely accurate** based *only* on the provided content.
        -   **Concise and to the point**, providing only the necessary information to answer the question.
        -   Free from external knowledge or hallucination.

        Return ONLY the JSON array with no additional text or formatting."""

    def format_content_batch(
        self, content_list: List[str], n_questions_per_content: int
    ) -> str:
        """Format content for batch processing."""
        formatted_text = "MEDICAL CONTENT FOR QUESTION GENERATION:\n\n"
        for i, content in enumerate(content_list, 1):
            formatted_text += f"CONTENT {i}:\n{content}\n\n"
        formatted_text += f"GENERATE {n_questions_per_content * len(content_list)} DISTINCT AND CHALLENGING QUESTIONS OF DIFFERENT TYPES FOR EACH CONTENT. FOR EACH QUESTION, ALSO PROVIDE A CONCISE AND ACCURATE GROUND TRUTH ANSWER THAT IS STRICTLY DERIVED FROM THE PROVIDED CONTENT ONLY."

        return formatted_text

    def generate_questions_batch(
        self, content_list: List[str], id_list: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate questions for a batch of content using OpenAI.

        Args:
            content_list: List of cleaned text content
            id_list: List of datapoint IDs corresponding to content pieces

        Returns:
            List of dictionaries with question, question_type, and source (ID)
        """
        try:
            prompt = self.format_content_batch(
                content_list, self.n_questions_per_content
            )

            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt[: self.max_length]},
                ],
                temperature=0,
                max_tokens=4000,
            )

            response_text = response.choices[0].message.content
            questions = json.loads(self.clean_json_response(response_text))

            # Post-process to add source IDs
            questions_with_source = self.add_source_to_questions(questions, id_list)

            logger.info(
                f"Generated {len(questions_with_source)} questions for batch of {len(content_list)} content pieces"
            )
            return questions_with_source

        except Exception as e:
            logger.error(f"Error generating questions for batch: {e}")
            return []

    def clean_json_response(self, response: str) -> str:
        """Clean the JSON response from OpenAI."""
        # Find JSON array bounds
        start_index = response.find("[")
        end_index = response.rfind("]")

        if start_index == -1 or end_index == -1:
            logger.warning("Could not find JSON array in response")
            return "[]"

        return response[start_index : end_index + 1]

    def add_source_to_questions(
        self, questions: List[Dict[str, Any]], id_list: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Post-process questions to add source datapoint IDs.

        Args:
            questions: List of questions from LLM (without source)
            id_list: List of datapoint IDs corresponding to content pieces

        Returns:
            List of questions with source ID added
        """
        questions_with_source = []

        # Calculate expected questions per content piece
        expected_total_questions = len(id_list) * self.n_questions_per_content

        if len(questions) != expected_total_questions:
            logger.warning(
                f"Expected {expected_total_questions} questions for {len(id_list)} content pieces, "
                f"but got {len(questions)}. Will map available questions to content pieces."
            )

        # Map questions to content pieces
        for i, question in enumerate(questions):
            # Determine which content piece this question belongs to
            content_index = i // self.n_questions_per_content

            # Handle case where we have more questions than expected
            if content_index >= len(id_list):
                content_index = len(id_list) - 1
                logger.warning(
                    f"Question {i} mapped to last content piece due to overflow"
                )

            # Add source ID to question
            question_with_source = question.copy()
            question_with_source["source_id"] = id_list[content_index]
            questions_with_source.append(question_with_source)

        return questions_with_source

    def fetch_all_cleaned_content(
        self, collection_name: str, chapter_names: List[str] = None
    ) -> tuple[List[str], List[str]]:
        """
        Fetch all cleaned content and their IDs from QdrantDB collection.

        Args:
            collection_name: Name of the QdrantDB collection
            chapter_names: List of chapter names to filter by (optional)

        Returns:
            Tuple of (content_list, id_list) where both lists have the same order
        """
        all_cleaned_contents = []
        all_content_ids = []
        query_filter = models.Filter(
            must=(
                [
                    models.FieldCondition(
                        key="chapter",
                        match=models.MatchAny(
                            any=chapter_names,
                        ),
                    )
                ]
                if chapter_names
                else None
            )
        )
        try:
            # Scroll through all points in the collection
            scroll_response = client.scroll(
                collection_name=collection_name, limit=10000, scroll_filter=query_filter
            )
            points = scroll_response[0]

            for point in points:
                cleaned_content = point.payload.get("cleaned_content")
                if (
                    cleaned_content
                    and isinstance(cleaned_content, str)
                    and len(cleaned_content.strip()) > 50
                ):
                    # get the cleaned content and the id
                    all_cleaned_contents.append(cleaned_content.strip())
                    all_content_ids.append(str(point.id))

            logger.info(
                f"Fetched {len(all_cleaned_contents)} cleaned content pieces from {collection_name}"
            )

        except Exception as e:
            logger.error(f"Error fetching content from {collection_name}: {e}")

        return all_cleaned_contents, all_content_ids

    def generate_evaluation_dataset(
        self,
        collection_name: str,
        batch_size: int = 2,
        filter_by_chapters: List[str] = None,
        output_filename: str = None,
    ) -> None:
        """
        Generate evaluation questions from cleaned content and upload to Comet.

        Args:
            collection_name: QdrantDB collection name containing cleaned content
            batch_size: Number of content pieces to process in each batch
            filter_by_chapters: List of chapter names to filter by (optional)
            output_filename: Custom filename for output (optional)
        """
        # Validate settings
        assert settings.COMET_API_KEY, "COMET_API_KEY must be set in settings"
        assert settings.COMET_WORKSPACE, "COMET_WORKSPACE must be set in settings"
        assert settings.COMET_PROJECT, "COMET_PROJECT must be set in settings"
        assert settings.OPENAI_API_KEY, "OPENAI_API_KEY must be set in settings"

        # Fetch all cleaned content and IDs
        cleaned_contents, content_ids = self.fetch_all_cleaned_content(
            collection_name, filter_by_chapters
        )

        if not cleaned_contents:
            logger.warning(f"No cleaned content found in collection {collection_name}")
            return

        logger.info(
            f"Processing {len(cleaned_contents)} content pieces from {collection_name}"
        )

        all_questions = []

        # Process content in batches
        for i in range(0, len(cleaned_contents), batch_size):
            batch_content = cleaned_contents[i : i + batch_size]
            batch_ids = content_ids[i : i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(cleaned_contents) + batch_size - 1) // batch_size}"
            )

            # Generate questions for this batch
            batch_questions = self.generate_questions_batch(batch_content, batch_ids)

            # Validate and add to results
            if (
                len(batch_questions)
                != len(batch_content) * self.n_questions_per_content
            ):
                logger.warning(
                    f"Expected {len(batch_content) * self.n_questions_per_content} questions, got {len(batch_questions)}. Continuing..."
                )

            all_questions.extend(batch_questions)

        logger.info(f"Generated {len(all_questions)} total questions")

        # Prepare output
        output_dir = Path("evaluation_dataset")
        output_dir.mkdir(exist_ok=True)

        filename = output_filename or f"{collection_name}_evaluation_questions.json"
        output_file = output_dir / filename

        # Save to JSON file
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(all_questions, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(all_questions)} questions to {output_file}")

        # Upload to Comet
        self.upload_to_comet(output_file, collection_name)

    def upload_to_comet(self, file_path: Path, collection_name: str) -> None:
        """
        Upload the evaluation dataset to Comet ML.

        Args:
            file_path: Path to the JSON file containing questions
            collection_name: Original collection name for artifact naming
        """
        try:
            logger.info(f"Uploading evaluation dataset to Comet: {file_path}")

            experiment = start()

            # Create artifact for the evaluation dataset
            artifact_name = f"{collection_name}-evaluation-dataset"
            artifact = Artifact(artifact_name, artifact_type="dataset")
            artifact.add(str(file_path))

            # Log the artifact
            experiment.log_artifact(artifact)
            experiment.end()

            logger.info(
                f"Successfully uploaded evaluation dataset to Comet as artifact '{artifact_name}'"
            )

        except Exception as e:
            logger.error(f"Failed to upload to Comet: {e}")


def main():
    """Main function to generate evaluation datasets."""
    generator = EvaluationQuestionGenerator(n_questions_per_content=5)

    # Define collections to process
    collections = [
        "cleaned_nice",  # NICE guidelines
    ]

    for collection_name in collections:
        try:
            logger.info(
                f"Starting evaluation dataset generation for collection: {collection_name}"
            )
            generator.generate_evaluation_dataset(
                collection_name=collection_name,
                batch_size=1,
                filter_by_chapters=[
                    "Overview",
                    "Recommendations",
                    "Recommendations for research",
                    "Rationale and impact",
                    "Context",
                ],
            )
            logger.info(
                f"Completed evaluation dataset generation for collection: {collection_name}"
            )
        except Exception as e:
            logger.error(f"Failed to process collection {collection_name}: {e}")


if __name__ == "__main__":
    main()
