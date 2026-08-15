"""
Command-Line Interface (CLI) Entrypoint for Training the 4 CNN Benchmark Models.
Supports model selection ('densenet121', 'inception_v3', 'inception_resnet_v2', 'custom_cnn', 'all'),
configurable epochs, batch size, learning rate, and smoke-test mode.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import (
    DEFAULT_IMAGE_SIZE, INCEPTION_IMAGE_SIZE, DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE, MAX_EPOCHS, RAW_PUBLIC_DATASET_DIR
)
from src.preprocessing.dataset_loader import get_dataloaders
from src.training.trainer import ExamModelTrainer
from src.utils.logger import get_logger

logger = get_logger("train_cli")

CNN_BENCHMARK_MODELS = [
    "custom_cnn",
    "densenet121",
    "inception_v3",
    "inception_resnet_v2"
]


def run_training_pipeline(
    model_name: str = "custom_cnn",
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    smoke_test: bool = False,
    resume: bool = True,
    dataset_dir: Path = RAW_PUBLIC_DATASET_DIR,
    weights_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Executes training pipeline for specified CNN model(s).
    """
    targets = CNN_BENCHMARK_MODELS if model_name.lower().strip() == "all" else [model_name.lower().strip()]
    results = {}

    for target_model in targets:
        logger.info(f"\n================ STARTING TRAINING PIPELINE: {target_model} ================")

        target_size = INCEPTION_IMAGE_SIZE if "inception" in target_model else DEFAULT_IMAGE_SIZE

        # Prepare DataLoaders
        train_loader, val_loader = get_dataloaders(
            dataset_dir=dataset_dir,
            target_size=target_size,
            batch_size=batch_size,
            max_samples_per_split=16 if smoke_test else None
        )

        kwargs = {
            "model_name": target_model,
            "pretrained": True if target_model != "custom_cnn" else False,
            "learning_rate": learning_rate
        }
        if weights_dir is not None:
            kwargs["weights_dir"] = weights_dir
        if results_dir is not None:
            kwargs["results_dir"] = results_dir

        trainer = ExamModelTrainer(**kwargs)

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1 if smoke_test else epochs,
            smoke_test=smoke_test,
            resume=resume
        )
        results[target_model] = history

        logger.info(f"================ FINISHED TRAINING PIPELINE: {target_model} ================\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train CNN Benchmark Models for Online Exam Proctoring.")
    parser.add_argument(
        "--model",
        type=str,
        default="custom_cnn",
        choices=["custom_cnn", "densenet121", "inception_v3", "inception_resnet_v2", "all"],
        help="Target model architecture to train (default: custom_cnn)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum training epochs (default: 30)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8 for VRAM safety)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Initial learning rate (default: 1e-4)."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Runs a fast 1-epoch / 2-batch smoke test validation without full training."
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Disable resuming from existing latest checkpoint."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=RAW_PUBLIC_DATASET_DIR,
        help="Path to verified public dataset directory."
    )

    args = parser.parse_args()

    results = run_training_pipeline(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        smoke_test=args.smoke_test,
        resume=args.resume,
        dataset_dir=args.dataset_dir
    )

    print("Training Pipeline Execution Complete.")
    for m, res in results.items():
        print(f"Model: {m} | Best Val Loss: {res.get('best_val_loss', 0.0):.4f} | Best Val Acc: {res.get('best_val_acc', 0.0):.4f}")


if __name__ == "__main__":
    main()
