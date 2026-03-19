from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import retrieve
from tavily import TavilyClient
import http.client
import re
import json
import logging
import os

log_level = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
logging.basicConfig(format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

AWS_REGION = os.getenv("AWS_REGION")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-lite-v1:0",
)

KB_PROMPT = """
Role:
Retrieve the items reference code, primary color, secondary color(s), primary outer material, and secondary outer material(s) to be used in search, and for final verification.

Guidelines:
Once the relevant information is obtained, return it.
If some information is classified as not available or to be updated, do not include it.
"""

SEARCH_PROMPT = """
Role:
Find current and past listings for items using web search.
Search using the query and the reference code and color retrieved from the knowledge base, combine them as one query.

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
You must pass all URLs from the search results into the validate_urls tool to help filter out any non-functional URLs.
Filter based on all of the ground truth provided by the knowledge base EXCEPT for the reference code. If a search result doesn't match the item based in the knowledge base, filter it out.
If a result is simply missing information that the knowledge base contains, DO NOT filter it out immediately. Treat "missing info" as a potential match unless it is proven wrong by other details.
Make sure retrieved results are from Dior Homme AW04.
Discard duplicate listings, replicas, inspired items, or unrelated pieces.
Provide listing URLs. Never guess a URL.
"""

@tool
def validate_urls(urls: list[str]) -> dict:
    """
    Validate and extract full content from a list of web URLs.

    Use this to verify if clothing listings are still active, and filter out dead links or error pages.

    Args:
    urls (list[str]): A list of URLs to be validated and extracted.

    Returns:
    A dictionary containing 'valid_listings' with extracted content and 'invalid' listings with failure reasons. Returns an error message if the extraction fails.
    """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = {"valid_listings": [], "invalid": []}
    
    pattern = re.compile(r".*listings.*|.*products.*|.*item.*", re.IGNORECASE)
    potential_urls = [u for u in urls if re.search(pattern, u)]

    try:
        extraction = tavily_client.extract(urls=potential_urls)
        for item in extraction.get("results", []):
            content = item.get("raw_content", "").lower()
            
            if "page not found" in content or "available" in content[:500]:
                results["invalid"].append({"url": item['url'], "reason": "Soft 404"})
                continue
            
            results["valid_listings"].append({
                "url": item['url'],
                "content": item['raw_content'][:3000],
                "title": item.get("title")
            })
    except Exception as e:
        return {"error": str(e)}

    return results

@tool
def serper_search(query: str) -> str:
    """
    Perform a web search for active listings, market data, or reference verification.

    Use this to search for clothing listings, reference codes, or prices on the web.

    Args:
    query (str): A search query.

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
    """
    Perform knowledge based filtered web search for active listings, market data, or reference verification.

    Use this to search for clothing listings, reference codes, or prices on the web, and have these results verified by knowledge base information.

    Args:
    query (str): A search query.

    Returns:
    Filtered search results.
    """
    kb_agent = Agent(model=bedrock_model,
        system_prompt=KB_PROMPT, tools=[retrieve])
    google_agent = Agent(model=bedrock_model,
        system_prompt=SEARCH_PROMPT, tools=[serper_search])
    aggregator_agent = Agent(model=bedrock_model,
        system_prompt=AGGREGATOR_PROMPT, tools=[validate_urls])

    kb_results = kb_agent(f"Retrieve relevant information based on this query: {query}")
    search_results = google_agent(f"Perform a web search based on the query and relevant knowledge base information. Query: {query}. Knowledge base results: {kb_results}.")
    response = aggregator_agent(f"Filter out any redundant or irrelevant results from these search results: {search_results}. Base the relevancy on these ground truths: {kb_results}.")
    return response