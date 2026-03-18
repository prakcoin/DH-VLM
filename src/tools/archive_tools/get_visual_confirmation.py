from strands import tool
import os
import boto3
from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import retrieve
from src.agents.hooks import LimitToolCounts

s3 = boto3.client('s3', region_name=os.getenv("AWS_REGION"))
bedrock = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION"))
BUCKET_NAME = 'aw04-data'
IMAGE_FOLDER = 'images/'
FOLDER_PREFIX = 'looks/'
CLOUDFRONT_DOMAIN = 'https://d39bzdkvoca64w.cloudfront.net'

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-2-lite-v1:0",
)

KB_PROMPT = """
Role:
Retrieve the look number, category, subcategory, primary and secondary color(s), pattern, primary and secondary outer material(s), and additional notes from the knowledge base based on the query.

Guidelines: 
If the look number is already included in the query, there is no need to retrieve it from the knowledge base.
Make sure the look number retrieved is a positive integer, and not a word or float. 
If retrieve returns no results or an error, return: INFO_NOT_AVAILABLE.
"""

VISUAL_PROMPT = """
Analyze look images for fit, silhouette, texture, and aesthetic details.

Guidelines:
Use the look number provided from the retrieved results to get the filenames using the get_look_images tool. 
Pass the retrieved image filenames into get_image_details in order to retrieve detailed visual analysis.
If get_look_images returns an empty list or an error, return: IMAGE_NOT_AVAILABLE.
"""

SYNTHESIS_PROMPT = """
Role:
Synthesize a final answer based on visual and knowledge base information.

Guidelines:
Combine visual analysis with metadata for the final answer.
Report discrepancies between visual and metadata observations.
"""

@tool 
def get_visual_confirmation(query: str) -> str:
    """
    Perform visual analysis based on a query, in order to confirm details not present in the knowledge base.

    Use this tool when a query requires direct visual inspection of garments, accessories, layering, closures, construction details, or physical attributes that cannot be reliably inferred from metadata alone. 

    Args:
    query (str): A specific visual question to answer.

    Returns:
    A structured textual analysis based only on confirmed visual observations.
    """
    limit_hook = LimitToolCounts(max_tool_counts={"retrieve": 3})

    kb_agent = Agent(model=bedrock_model,
        system_prompt=KB_PROMPT, tools=[retrieve], hooks=[limit_hook])
    visual_agent = Agent(model=bedrock_model,
        system_prompt=VISUAL_PROMPT, tools=[get_look_images, get_image_details])
    synthesis_agent = Agent(model=bedrock_model,
        system_prompt=SYNTHESIS_PROMPT)

    kb_results = kb_agent(f"Retrieve the look number based on this query: {query}")
    if not str(kb_results).strip():
        return "The retrieval system failed to return a response. Please try again."
    if "info_not_available" in str(kb_results).lower():
        return f"I'm sorry, I couldn't find a specific look in our records for '{query}'."
    visual_results = visual_agent(f"Based on the look number retrieved, answer the query. Retrived results: {str(kb_results)}. Query: {query}.")
    if not str(visual_results).strip():
        return "The visual analysis system failed to return a response. Please try again."
    if "image_not_available" in str(visual_results).lower():
        return f"I found the metadata for '{query}', but no archival images were available for visual analysis."
    response = synthesis_agent(f"Synthesize a final result for this query: {query}. Visual results: {str(visual_results)}. Knowledge base results: {str(kb_results)}.")
    return response