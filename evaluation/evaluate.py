import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.utils import create_dataset, create_evaluators, run_async_evaluation, get_multiturn_response, get_response
import asyncio
import logging

logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

if __name__ == "__main__":
    mode = "search"

    response_fn = get_multiturn_response if mode == "followups" else get_response

    test_cases = create_dataset(mode=mode)
    evaluators = create_evaluators()
    report = asyncio.run(run_async_evaluation(mode, test_cases, evaluators, response_fn))