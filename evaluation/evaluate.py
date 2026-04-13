import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.utils import create_dataset, run_async_evaluation, run_evaluation
import asyncio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent evaluations.")
    parser.add_argument("--mode", "-m", required=True, help="Evaluation dataset to run (e.g. aggregation, general, followups).")
    args = parser.parse_args()

    test_cases = create_dataset(mode=args.mode)

    if args.mode == "followups":
        run_evaluation(args.mode, test_cases)
    else:
        asyncio.run(run_async_evaluation(args.mode, test_cases))