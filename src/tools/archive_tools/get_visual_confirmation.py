from strands import tool
import os
import boto3
import json
import base64
from urllib.parse import urlparse
import os
from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import retrieve, stop
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
Retrieve the look number, category, subcategory, primary and secondary color(s), pattern, primary and secondary outer material(s), and additional notes from the knowledge base based on the query.

Guidelines: 
Use the retrieve tool to get the relevant information, then return it.
If retrieve returns no results or an error, use the stop tool with reason INFO_NOT_AVAILABLE.
If the look number is already included in the query, there is no need to retrieve it from the knowledge base.
Make sure the look number retrieved is a positive integer, and not a word or float. 
"""

VISUAL_PROMPT = """
Analyze look images for fit, silhouette, texture, and aesthetic details.

Guidelines:
Use the look number provided from the retrieved results to get the filenames using the get_look_images tool. 
Pass the retrieved image filenames into get_image_details in order to retrieve detailed visual analysis.
If get_look_images returns an empty list or an error, use the stop tool with reason IMAGE_NOT_AVAILABLE.
"""

SYNTHESIS_PROMPT = """
Role:
Synthesize a final answer based on visual and knowledge base information.

Guidelines:
Combine visual analysis with metadata for the final answer.
Report discrepancies between visual and metadata observations.
"""

@tool
def get_look_images(look_number: str):
    """
    Retrieve the runway images for a specific look.
    
    Use this tool when a user asks to see a specific runway look. Only use it when you have a look number.
    
    Args:
    look_number (str): The unique identifier for the look, e.g., "1".

    Returns: 
    A list of image URLs for the look.
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
    
    return image_urls

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
        system_prompt=KB_PROMPT, tools=[retrieve, stop], hooks=[limit_hook])
    visual_agent = Agent(model=bedrock_model,
        system_prompt=VISUAL_PROMPT, tools=[get_look_images, get_image_details, stop])
    synthesis_agent = Agent(model=bedrock_model,
        system_prompt=SYNTHESIS_PROMPT)

    kb_results = kb_agent(f"Retrieve the look number based on this query: "
                          f"Query: {query}.")
    if not str(kb_results).strip():
        return "No matching look number found in the knowledge base."
    visual_results = visual_agent(f"Answer the query based on the retrieved look number. "
                                  f"Query: {query}. "
                                  f"Retrieved results: {str(kb_results)}.")
    if not str(visual_results).strip():
        return "Visual analysis for this look is currently unavailable."
    response = synthesis_agent(f"Synthesize a final result for the query based on the visual and knowledge base results. "
                               f"Query {query}. "
                               f"Visual results: {str(visual_results)}. " 
                               f"Knowledge base results: {str(kb_results)}.")
    return response