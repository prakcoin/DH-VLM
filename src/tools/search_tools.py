from strands import tool
import http.client
import json
import logging
import os
import urllib.request
from dotenv import load_dotenv

load_dotenv()

log_level = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
logging.basicConfig(format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

@tool
def serper_search(query: str) -> str:
    """
    Perform a web search for active listings, market data, or reference verification.

    Use this to search for clothing listings, reference codes, or prices on the web.

    Args:
    query (str): A search query for Serper search.

    Returns:
    Raw JSON search results from the search API. 
    Return "Search failed." if the request is unsuccessful.
    """
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query, "num": 5}) 
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        return data
    except Exception as e:
        logger.error(f"Serper request failed: {e}")
        return "Search failed."

@tool
def tavily_search(query: str) -> str:
    """
    Perform a web search for historical, contextual, or analytical information.

    Use this tool when deeper research is required, such as collection history, runway analysis, design inspirations, or editorial commentary.

    Args:
    query (str): A search query for Tavily search.

    Returns:
    Raw JSON search results from the search API. 
    Return "Research failed." if the request is unsuccessful.
    """
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "max_results": 3
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        logger.error(f"Tavily request failed: {e}")
        return "Research failed."