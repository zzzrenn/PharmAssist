import argparse
import os
import sys
from pathlib import Path

# Add the project root to path to resolve module imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import settings
from huggingface_hub import HfApi
from sagemaker.huggingface import HuggingFace

from core import logger_utils

logger = logger_utils.get_logger(__file__)

finetuning_dir = Path(__file__).resolve().parent
finetuning_requirements_path = finetuning_dir / "requirements.txt"


def run_finetuning_on_sagemaker(
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    is_dummy: bool = False,
) -> None:
    assert settings.HUGGINGFACE_ACCESS_TOKEN, (
        "Hugging Face access token (HUGGINGFACE_ACCESS_TOKEN) is required. Update your .env file."
    )
    assert settings.AWS_ARN_ROLE, (
        "AWS ARN role (AWS_ARN_ROLE) is required. Update your .env file."
    )
    assert settings.COMET_API_KEY, (
        "Comet ML API key (COMET_API_KEY) is required. Update your .env file."
    )
    assert settings.COMET_WORKSPACE, (
        "Comet ML workspace (COMET_WORKSPACE) is required. Update your .env file."
    )
    assert settings.COMET_PROJECT, (
        "Comet ML project name (COMET_PROJECT) is required. Update your .env file."
    )

    if not finetuning_dir.exists():
        raise FileNotFoundError(f"The directory {finetuning_dir} does not exist.")
    if not finetuning_requirements_path.exists():
        raise FileNotFoundError(
            f"The file {finetuning_requirements_path} does not exist."
        )

    api = HfApi()
    user_info = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    huggingface_user = user_info["name"]
    logger.info(f"Current Hugging Face user: {huggingface_user}")

    hyperparameters = {
        "base_model_name": settings.HUGGINGFACE_BASE_MODEL_ID,
        "dataset_id": settings.DATASET_ID,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "model_output_huggingface_workspace": huggingface_user,
    }
    if is_dummy:
        hyperparameters["is_dummy"] = True

    logger.info(f"Training pipeline with hyperparameters: {hyperparameters}")

    # Create the HuggingFace SageMaker estimator
    huggingface_estimator = HuggingFace(
        entry_point="finetune.py",
        source_dir=str(finetuning_dir),
        instance_type="ml.g5.2xlarge",
        instance_count=1,
        role=settings.AWS_ARN_ROLE,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        hyperparameters=hyperparameters,
        requirements_file=finetuning_requirements_path,
        environment={
            "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
            "COMET_API_KEY": settings.COMET_API_KEY,
            "COMET_WORKSPACE": settings.COMET_WORKSPACE,
            "COMET_PROJECT_NAME": settings.COMET_PROJECT,
        },
    )

    # Start the training job on SageMaker.
    huggingface_estimator.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is-dummy", action="store_true", help="Run in dummy mode")
    args = parser.parse_args()

    logger.info(f"Is the training pipeline in DUMMY mode? '{args.is_dummy}'")

    run_finetuning_on_sagemaker(is_dummy=args.is_dummy)
