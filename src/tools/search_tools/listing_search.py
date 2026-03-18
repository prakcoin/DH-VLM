from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import retrieve
import http.client
import json
import logging
import os

log_level = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
logging.basicConfig(format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

AWS_REGION = os.getenv("AWS_REGION")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-lite-v1:0",
)

KB_PROMPT = """
Role:
Retrieve proper information to be used in search, and for final verification.

Guidelines:
Once the relevant information is obtained, add it to the rest of the query. 
The only required information to add is the item name.
If some information is classified as not available or to be updated, do not include it.
"""

SEARCH_PROMPT = """
Role:
Find current and past listings for items using web search.

Guidelines:
Limit searches strictly to Dior Homme Autumn/Winter 2004, by Hedi Slimane. Avoid Dior by John Galliano, Christian Dior, Christian Dior Monsieur, or other seasons/era collections.
Always include season and collection identifiers if the user query is vague.
Discard replicas, inspired items, or unrelated pieces.
Include source URLs with every fact. Do not hallucinate invalid URLs.
"""

AGGREGATOR_PROMPT = """
Role:
Aggregate all web search results and filter out irrelevant or redundant results.

Guidelines:
Filter based on ground truth provided by the knowledge base agent. If a search result doesn't match the item based in the knowledge base, filter it out. 
Make sure retrieved results are from Dior Homme AW04.
Discard duplicate listings, replicas, inspired items, or unrelated pieces.
Provide listing URLs. If a URL cannot be verified as functional, discard it. Never guess a URL.
"""

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
    search_query = f"Dior Homme AW04 {query}"
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": search_query, "num": 5}) 
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
def listing_search(query: str) -> str:
    kb_agent = Agent(model=bedrock_model,
        system_prompt=KB_PROMPT, tools=[retrieve])
    google_agent = Agent(model=bedrock_model,
        system_prompt=SEARCH_PROMPT, tools=[serper_search])
    aggregator_agent = Agent(model=bedrock_model,
        system_prompt=AGGREGATOR_PROMPT)

    kb_results = kb_agent(f"Retrieve relevant information based on this query: {query}")
    search_results = google_agent(f"{kb_results}")
    response = aggregator_agent(f"Filter out any redundant or irrelevant results from these search results: {search_results}. Base the relevancy on these ground truths: {kb_results}")
    return response