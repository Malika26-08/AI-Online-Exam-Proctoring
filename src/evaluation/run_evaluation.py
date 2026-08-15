"""
CLI Entry Point for Running Model Evaluation and Benchmarking Phase.
Evaluates saved CNN checkpoints on the labeled validation split,
plots loss/acc training curves, exports metrics to results/, and prints summary table.
"""

import sys
import pandas as pd
from src.evaluation.benchmark import ModelBenchmarker
from src.utils.logger import get_logger

logger = get_logger("run_evaluation")


def main():
    logger.info("Executing Phase 4 Model Evaluation and Benchmarking Pipeline...")
    benchmarker = ModelBenchmarker()
    df = benchmarker.run_benchmark()

    print("\n" + "=" * 90)
    print("FINAL CNN MODEL BENCHMARK COMPARISON TABLE (VALIDATION SPLIT ONLY)")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")
    logger.info("Evaluation and benchmarking pipeline completed successfully.")


if __name__ == "__main__":
    main()
