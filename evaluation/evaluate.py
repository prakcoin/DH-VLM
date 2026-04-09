import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.utils import create_dataset, run_async_evaluation, run_evaluation
import asyncio

if __name__ == "__main__":
    mode = "followups"

    test_cases = create_dataset(mode=mode)

    if mode == "followups":
        run_evaluation(mode, test_cases)
    else:
        asyncio.run(run_async_evaluation(mode, test_cases))