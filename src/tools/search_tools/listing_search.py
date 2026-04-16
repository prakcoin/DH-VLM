from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import retrieve, stop
from tavily import TavilyClient
from src.agents.hooks import LimitToolCounts
from src.agents.handlers import AgentSteeringHandler
import urllib.request
import json
import logging
import os

log_level = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
logging.basicConfig(format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-2-lite-v1:0",
    temperature=0.0,
    max_tokens=12000
)

KB_PROMPT = """
Role:
Retrieve the items reference code, primary color, secondary color(s), primary outer material, secondary outer material(s), and additional notes to be used in search, and for final verification.
Use the retrieve tool to get the relevant information, then return it.
If retrieve returns no results or an error, use the stop tool with reason INFO_NOT_AVAILABLE.
"""

kb_handler = AgentSteeringHandler(
    system_prompt="""
    You are providing guidance to ensure proper formatting of information.

    Guidance:
    Make sure reference code, primary and secondary color(s), primary and secondary outer material(s), and additional notes are retrieved.
    If some information is classified as not available or to be updated, do not include it.
    
    When the tools return their responses, evaluate the text and deliver the final response directly to the user.
    """
)

SEARCH_PROMPT = """
Role:
Find current and past listings for items using the tavily_search tool.
If the search returns no results, use the stop tool with reason RESULTS_NOT_AVAILABLE.
"""

search_handler = AgentSteeringHandler(
    system_prompt="""
    You are providing guidance to ensure proper formatting of information.

    Guidance: 
    Make sure a list of search results is retrieved, and that the output is not off topic.
    
    When the tools return their responses, evaluate the text and deliver the final response directly to the user.
    """
)

AGGREGATOR_PROMPT = """
Role:
Aggregate all web search results and filter out irrelevant or redundant results.
You must pass all URLs from the search results into the validate_urls tool to help filter out any non-functional URLs.
"""

aggregator_handler = AgentSteeringHandler(
    system_prompt="""
    You are providing guidance to ensure proper formatting of information.

    Guidance:
    Filter based on all of the ground truth provided by the knowledge base. If a known reference code is present in a listing, treat it as a strong positive match. If a search result doesn't match the item based on the knowledge base, filter it out.
    If a result is simply missing information that the knowledge base contains, DO NOT filter it out immediately. Treat "missing info" as a potential match unless it is proven wrong by other details.
    Discard replicas, inspired items, or unrelated pieces.
    Provide listing URLs. If no URLs are provided, skip validation and state that no results were found.
    If there are no relevant results, and no results match what is being asked, then state this and return no results.
        
    When the tools return their responses, evaluate the text and deliver the final response directly to the user.
    """
)

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
                "content": content[:6000],
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
            "domains": ["grailed.com", "ebay.com", "vestiairecollective.com", "therealreal.com", "zentmpl.com", "1stdibs.com"],
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

    payload = {
        "api_key": api_key,
        "query": f"Dior {query}",
        "include_images": True,
        "country": "united states",
        "search_depth": "advanced",
        "max_results": 3
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            results.extend(res_data.get("results", []))
    except Exception as e:
        logger.error(f"Search failed for united states variant 'Google': {e}")


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
                with urllib.request.urlopen(req, timeout=30) as response:
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
    limit_validate_hook = LimitToolCounts(max_tool_counts={"validate_urls": 3})

    kb_agent = Agent(model=bedrock_model,
        system_prompt=KB_PROMPT, tools=[retrieve, stop], hooks=[limit_retrieve_hook], plugins=[kb_handler], callback_handler=None)
    google_agent = Agent(model=bedrock_model,
        system_prompt=SEARCH_PROMPT, tools=[tavily_search, stop], hooks=[limit_search_hook], plugins=[search_handler], callback_handler=None)
    aggregator_agent = Agent(model=bedrock_model,
        system_prompt=AGGREGATOR_PROMPT, tools=[validate_urls], hooks=[limit_validate_hook], plugins=[aggregator_handler], callback_handler=None)

    kb_results = kb_agent(f"Retrieve relevant information based on this query. " 
                          f"Query: {query}")
    if not str(kb_results).strip():
        return "No matching AW04 metadata found in the knowledge base."
    search_results = google_agent(f"Perform a web search based on the query and relevant knowledge base information. "
                                  f"Query: {query}. " 
                                  f"Knowledge base results: {str(kb_results)}.")
    if not str(search_results).strip():
        return "No Dior Homme AW04 listings were found matching your criteria."
    response = aggregator_agent(f"Filter out any irrelevant results from the search results, basing the relevancy on the query and knowledge base results. " 
                                f"Search results: {str(search_results)}. " 
                                f"Query: {query}. "
                                f"Knowledge base results: {str(kb_results)}.")
    return response