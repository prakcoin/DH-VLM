from strands import tool
import os
import boto3
import json
import base64
from urllib.parse import urlparse
import csv
import io
import logging

from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import retrieve, stop
from src.agents.hooks import LimitToolCounts
from strands.vended_plugins.steering import LLMSteeringHandler
from src.agents.handlers import ModelOutputSteeringHandler, ToolInputSteeringHandler

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3', region_name=os.getenv("AWS_REGION"))
bedrock = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION"))
BUCKET_NAME = 'aw04-data'
IMAGE_FOLDER = 'images/'
FOLDER_PREFIX = 'looks/'
CLOUDFRONT_DOMAIN = 'https://d39bzdkvoca64w.cloudfront.net'

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-2-lite-v1:0",
)

DETAIL_PROMPT = """
Role:
Perform grounded, multi-image visual analysis focusing on technical construction and garment specifications.

Guidelines:
Use all provided images collectively. If a detail is visible in only one image, it is present. In case of conflict, prioritize the highest-resolution view.
Step 1- List all visible items (top to bottom) including Type, Color (single word), Construction details, Position, and Hardware. Use "unclear due to resolution" for uncertain details.
Step 2- Perform a focused scan of the following: 
- Jewelry: Scan wrists, fingers, and neck.
- Layers: Count neckline and sleeve layers separately.
- Closures: Describe the mechanism before naming it.
- Lapels: Describe the geometry/shape before classifying.
- Hems: Describe folds, stacking, or raw edge appearance.
Step 3- Respond using ONLY confirmed observations. Do not infer brand, season, or intent. If evidence is insufficient, state it clearly.
"""

KB_PROMPT = """
Role:
Retrieve the look number, category, subcategory, primary and secondary color(s), pattern, primary and secondary outer material(s), and additional notes from the knowledge base based on the query using the retrieve tool. 
Retrieve the look composition using the look number (either retrieved from the knowledge base, or provided by the user) and pass it in the get_look_composition tool. 
If retrieve returns no results or an error, use the stop tool with reason INFO_NOT_AVAILABLE.
If there is no look number provided, retrieve it using the query and the retrieve tool. 
If the look number is already included in the query, there is no need to retrieve it from the knowledge base.
"""

kb_tool_handler = ToolInputSteeringHandler(
    # system_prompt="""
    # You are providing guidance to ensure proper formatting of tool inputs.

    # Guidance:
    # Do not retrieve using the full query, instead extract the core subject (e.g., "leather jacket") and search with this instead.
    # Using the retrieved look number, retrieve the look composition by passing the look number into the get_look_composition tool.    
    # """
)

kb_handler = ModelOutputSteeringHandler(
    system_prompt="""
    You are providing guidance to ensure proper formatting of information.

    Guidance:
    Make sure look number, category, subcategory, primary and secondary color(s), pattern, primary and secondary outer material(s), and additional notes are retrieved.
    If some information is classified as not available or to be updated, do not include it.
    Make sure the look number retrieved is a positive integer, and not a word or float.
    The output must contain both the retrieved knowledge base metadata and the look composition.
    
    When the tools return their responses, evaluate the text and deliver the final response directly to the user.
    """
)


VISUAL_PROMPT = """
Role:
Analyze look images for fit, silhouette, texture, and aesthetic details.
"""

visual_tool_handler = ToolInputSteeringHandler(
    # system_prompt="""
    # You are providing guidance to ensure proper formatting of tool inputs.

    # Guidance:
    # Use the retrieved image filenames and pass them into get_image_details in order to retrieve detailed visual analysis.
    # """
)

# visual_handler = ModelOutputSteeringHandler(
#     system_prompt="""
#     You are providing guidance to ensure proper formatting of information.

#     Guidance:
#     Ensure a detailed visual analysis is provided.

#     When the tools return their responses, evaluate the text and deliver the final response directly to the user.
#     """
# )

SYNTHESIS_PROMPT = """
Role:
Synthesize a final answer based on visual and knowledge base information.
Combine visual analysis with metadata for the final answer.
Report discrepancies between visual and metadata observations.
"""

# synthesis_handler = ModelOutputSteeringHandler(
#     system_prompt="""
#     You are providing guidance to ensure proper formatting of information.

#     Guidance:
#     Determine if the results are conclusive. If they aren't, state this.
#     Ensure a fully synthesized answer is provided.
    
#     When the tools return their responses, evaluate the text and deliver the final response directly to the user.
#     """
# )


@tool
def get_look_composition(look_number: str):
    """
    Retrieve archival composition data and the runway images for a specific look.
    
    Use this tool when you want to list every item included in a specific runway look, or when a user asks to see a specific runway look. Only use it when you have a look number.
    
    Args:
    look_number (str): The unique identifier for the look, e.g., "1".

    Returns: 
    A list of every item in the requested look, and a list of image URLs for the look.
    """
    prefix = f"{IMAGE_FOLDER}look{look_number}_"
    image_objects = s3.list_objects_v2(
        Bucket=BUCKET_NAME, 
        Prefix=prefix,
    )
    
    image_urls = []
    if 'Contents' in image_objects:
        for obj in image_objects['Contents']:
            key = obj['Key']
            if key.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_url = f"{CLOUDFRONT_DOMAIN}/{key}"
                image_urls.append(full_url)
    
    clean_id = str(look_number).strip().lower().replace('look', '').strip()
    target_file = f"{FOLDER_PREFIX}look_{clean_id}.csv"
    
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=target_file)
        content = response['Body'].read().decode('utf-8-sig')
        data = list(csv.DictReader(io.StringIO(content)))
        
        report = f"Look {look_number} Composition:\n"
        report += f"Items: {str(data)}\n"
        report += f"Images: {', '.join(image_urls) if image_urls else 'No images found.'}"
        return report

    except s3.exceptions.NoSuchKey:
        logger.error(f"File not found: {target_file}")
        return f"I'm sorry, I couldn't find archival data for Look {look_number}."
    except Exception as e:
        logger.error(f"Error fetching {target_file}: {str(e)}")
        return f"Error retrieving data for Look {look_number}."

def parse_filenames_from_string(filenames_str):
    s = filenames_str.strip().lstrip("[").rstrip("]")
    parts = s.split(",")
    urls = [p.strip().strip('"').strip("'") for p in parts if p.strip()]
    return urls

@tool
def get_image_details(image_filenames, query: str):
    """
    Perform grounded visual analysis on one or more look images.

    Use this tool when a query requires direct visual inspection of garments, accessories, layering, closures, construction details, or physical attributes that cannot be reliably inferred from metadata alone. 

    Args:
    image_filenames (list): One or more image filenames or URLs associated with a look.
    query (str): A specific visual question to answer.

    Returns:
    A structured textual analysis based only on confirmed visual observations.
    """
    try:
        if not image_filenames:
            return "Error: No image filenames provided."

        if isinstance(image_filenames, str):
            image_filenames = parse_filenames_from_string(image_filenames)

        content_blocks = []

        for filename in image_filenames:
            parsed = urlparse(filename)
            clean_filename = os.path.basename(parsed.path)
            image_key = f"images/{clean_filename}"

            response = s3.get_object(Bucket=BUCKET_NAME, Key=image_key)
            image_bytes = response['Body'].read()

            content_blocks.append({
                "image": {
                    "format": "jpeg",
                    "source": {
                        "bytes": base64.b64encode(image_bytes).decode("utf-8")
                    }
                }
            })

        content_blocks.append({
            "text": DETAIL_PROMPT
        })
        content_blocks.append({
            "text": f"Query: {query}"
        })

        body = json.dumps({
            "inferenceConfig": {
                "max_new_tokens": 700,
                "temperature": 0.0
            },
            "messages": [
                {
                    "role": "user",
                    "content": content_blocks
                }
            ]
        })

        response = bedrock.invoke_model(
            modelId="amazon.nova-pro-v1:0",
            body=body
        )

        response_body = json.loads(response.get("body").read())
        return response_body["output"]["message"]["content"][0]["text"]

    except Exception as e:
        return f"Error analyzing images {image_filenames}: {str(e)}"

@tool 
def get_look_analysis(query: str) -> str:
    """
    Perform look analysis based on a query.

    Use this tool when a query requires a look breakdown, visual analysis of a look, or general questions about a look. 

    Args:
    query (str): A specific question about a look to answer.

    Returns:
    A structured textual analysis.
    """
    limit_hook = LimitToolCounts(max_tool_counts={"retrieve": 3})

    kb_agent = Agent(model=bedrock_model,
        system_prompt=KB_PROMPT, tools=[retrieve, get_look_composition, stop], hooks=[limit_hook], plugins=[kb_handler, kb_tool_handler])
    visual_agent = Agent(model=bedrock_model,
        system_prompt=VISUAL_PROMPT, tools=[get_image_details, stop], plugins=[visual_tool_handler])
    synthesis_agent = Agent(model=bedrock_model,
        system_prompt=SYNTHESIS_PROMPT)

    kb_results = kb_agent(f"Retrieve the look number and composition based on this query: "
                          f"Query: {query}.")
    if not str(kb_results).strip():
        return "No matching look number found in the knowledge base."
    visual_results = visual_agent(f"Answer the query based on the retrieved look number and composition. "
                                  f"Query: {query}. "
                                  f"Retrieved results: {str(kb_results)}.")
    if not str(visual_results).strip():
        return "Visual analysis for this look is currently unavailable."
    response = synthesis_agent(f"Synthesize a final result for the query based on the visual and textual results. "
                               f"Query {query}. "
                               f"Visual results: {str(visual_results)}. " 
                               f"Textual results: {str(kb_results)}.")
    return response