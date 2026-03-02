from strands import Agent, tool
from strands.models import BedrockModel
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

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-lite-v1:0",
)

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

SEARCH_PROMPT = """
Role:
Provide verified market and historical context via web searches.

Toolset / Actions:
serper_search: Active or historical item listings, secondary market prices, reference verification.
tavily_search: Deep-dive research into collection history, inspirations, show details.

Guidelines:
Limit searches strictly to Dior Homme AW04.
Always include season and collection identifiers if the user query is vague.
Discard replicas, inspired items, or unrelated pieces.
Include source URLs with every fact.
"""

@tool
def search_assistant(query: str) -> str:
    """
    Handle queries requiring web search.

    Use this as a conversational agent for performing web searches for details unseen in the metadata.

    Args:
    query (str): A question requiring external web search.

    Returns:
    Textual response synthesizing information from web sources, including cited URLs where applicable.
    """
    try:
        search_agent = Agent(
            model=bedrock_model,
            system_prompt=SEARCH_PROMPT,
            tools=[serper_search, tavily_search]
        )

        response = search_agent(query)
        return str(response)
    except Exception as e:
        return f"Error in search assistant: {str(e)}"