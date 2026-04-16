from strands import tool
import json
import logging
import os
import urllib.request

log_level = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
logging.basicConfig(format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

AWS_REGION = os.getenv("AWS_REGION")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

@tool
def general_search(query: str) -> str:
    """
    Perform a web search for historical, contextual, or analytical information.

    Use this tool when deeper research is required, such as collection history, design inspirations, or editorial commentary.

    Args:
    query (str): A search query.

    Returns:
    Raw JSON search results from the search API. 
    Return "Research failed." if the request is unsuccessful.
    """
    collection_context = "Dior Homme AW04"
    enriched_query = f"{collection_context} {query}"

    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": enriched_query,
        "search_depth": "advanced",
        "max_results": 5
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        logger.error(f"Tavily request failed: {e}")
        return "Research failed."