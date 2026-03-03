from strands import Agent, tool
from strands.models import BedrockModel
from src.tools.aggregation_tools import get_collection_items, get_collection_summary, get_item_counts
import boto3
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="us-east-1"
)

BUCKET_NAME = 'aw04-data'
FOLDER_PREFIX = 'looks/'

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-lite-v1:0",
)


AGGREGATION_PROMPT = """
Role:
Answer questions that require aggregation or analysis across the entire collection.

Guidelines:
Exclude generic functional components (buttons, belts, solids) unless explicitly asked.
Consolidate duplicate entries.
"""

@tool
def aggregation_assistant(query: str) -> str:
    """
    Handle collection-wide queries about multiple items, looks, and their metadata.

    Use this as a conversational agent to answer questions about collection-wide queries involving counts, recurring motifs, and specific information.

    Args:
    query (str): A question about aggregation.

    Returns: 
    Textual response with aggregated information.
    """
    try:
        aggregation_agent = Agent(
            model=bedrock_model,
            system_prompt=AGGREGATION_PROMPT,
            tools=[get_collection_items, get_collection_summary, get_item_counts]
        )

        response = aggregation_agent(query)
        return str(response)
    except Exception as e:
        return f"Error in aggregation assistant: {str(e)}"