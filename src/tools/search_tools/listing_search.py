from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import retrieve
from tavily import TavilyClient
from src.agents.hooks import LimitToolCounts
import urllib.request
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
Retrieve the items reference code, primary color, secondary color(s), primary outer material, secondary outer material(s), and additional notes to be used in search, and for final verification.

Guidelines:
Do not retrieve using the full query, instead extract the core subject (e.g., "leather jacket") and search with this instead.
Use the retrieve tool to get the relevant information, then return it.
If some information is classified as not available or to be updated, do not include it.
"""

SEARCH_PROMPT = """
Role:
Find current and past listings for items using the tavily_search tool.

Guidelines:
Search using the query and the reference code and color retrieved from the knowledge base, combine them as one query.
Do not search using the raw user query. Extract the core subject (e.g., "fur hooded jacket") and merge it with the retrieved knowledge base metadata. Example: input = "Can you find listings of the fur hooded jacket" + knowledge base "black, 4HH5043801" = black fur hooded leather jacket 4HH5043801.
Do not include the brand name or season in your input to avoid redundancy, as the tavily_search tool automatically applies hardcoded prefixes to search queries. Do not include "Dior", "AW04", "Autumn/Winter 2004", or similar keywords.
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
If no URLs are found, skip validation and state that no results were found.
Filter based on all of the ground truth provided by the knowledge base EXCEPT for the reference code. If a search result doesn't match the item based in the knowledge base, filter it out.
If a result is simply missing information that the knowledge base contains, DO NOT filter it out immediately. Treat "missing info" as a potential match unless it is proven wrong by other details.
Make sure retrieved results are from Dior Homme Autumn/Winter 2004.
Discard replicas, inspired items, or unrelated pieces.
Provide listing URLs. Never guess a URL.
If there are no relevant results, and no results match what is being asked, then state this and return no results.
"""

@tool
def validate_urls(urls: list[str]) -> dict:
    """
    Validate a list of web URLs.

    Use this to verify if clothing listings are still active, and filter out dead links or error pages.

    Args:
    urls (list[str]): A list of URLs to be validated and extracted.

    Returns:
    A dictionary containing 'valid_listings' with extracted content and 'invalid' listings with failure reasons. Returns an error message if the extraction fails.
    """
    if not urls:
        return {"valid_listings": [], "invalid": [], "message": "No URLs provided to validate"}

    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = {"valid_listings": [], "invalid": []}
    
    try:
        extraction = tavily_client.extract(urls=urls)
        
        for failed in extraction.get("failed_results", []):
            u = failed.get("url")
            results["invalid"].append({"url": u, "reason": "Hard 404"})

        for item in extraction.get("results", []):
            content = item.get("raw_content", "")
        
            results["valid_listings"].append({
                "url": item['url'],
                "content": content[:3500],
                "title": item.get("title")
            })
            
    except Exception as e:
        return {"error": f"Extraction failed: {str(e)}"}

    return results

@tool
def tavily_search(query: str) -> str:
    """
    Perform a web search for active listings, market data, or reference verification.

    Use this to search for clothing listings, reference codes, or prices on the web.

    Args:
    query (str): A search query.

    Returns:
    Raw JSON search results from the search API. 
    Return "Search failed." if the request is unsuccessful.
    """
    api_key = TAVILY_API_KEY
    url = "https://api.tavily.com/search"
    
    regions = [
        {
            "country": "united states",
            "domains": ["grailed.com", "ebay.com", "vestiairecollective.com", "therealreal.com"],
            "query_variants": [
                f"Dior {query}", 
                f"Dior AW04 {query}", 
                f"Dior Homme {query}", 
                f"Dior Homme AW04 {query}"
            ]
        },
        {
            "country": "japan",
            "domains": ["auctions.yahoo.co.jp", "jp.mercari.com", "fril.jp", "trefac.jp"],
            "query_variants": [
                f"ディオール {query}", 
                f"ディオール 04AW {query}", 
                f"ディオールオム {query}", 
                f"ディオールオム 04AW {query}"
            ]
        }
    ]

    results = []

    for region in regions:
        for variant in region["query_variants"]:
            payload = {
                "api_key": api_key,
                "query": variant,
                "include_images": True,
                "country": region["country"],
                "include_domains": region["domains"],
                "search_depth": "advanced",
                "max_results": 3
            }

            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    results.extend(res_data.get("results", []))
            except Exception as e:
                logger.error(f"Search failed for {region['country']} variant '{variant}': {e}")

    if not results:
        return "No results found."

    unique_results = []
    seen_urls = set()

    for result in results:
        url_link = result.get("url")
        if url_link not in seen_urls:
            unique_results.append(result)
            seen_urls.add(url_link)

    return json.dumps(unique_results, ensure_ascii=False)

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
    limit_retrieve_hook = LimitToolCounts(max_tool_counts={"retrieve": 3})
    limit_search_hook = LimitToolCounts(max_tool_counts={"tavily_search": 3})

    kb_agent = Agent(model=bedrock_model,
        system_prompt=KB_PROMPT, tools=[retrieve], hooks=[limit_retrieve_hook])
    google_agent = Agent(model=bedrock_model,
        system_prompt=SEARCH_PROMPT, tools=[tavily_search], hooks=[limit_search_hook])
    aggregator_agent = Agent(model=bedrock_model,
        system_prompt=AGGREGATOR_PROMPT, tools=[validate_urls])

    kb_results = kb_agent(f"Retrieve relevant information based on this query: {query}")
    search_results = google_agent(f"Perform a web search based on the query and relevant knowledge base information. Query: {query}. Knowledge base results: {str(kb_results)}.")
    if "No results found" in str(search_results):
        return "No Dior Homme AW04 listings were found matching your criteria."
    response = aggregator_agent(f"Filter out any irrelevant results from these search results: {str(search_results)}. Base the relevancy on these ground truths: {str(kb_results)}.")
    return response