from strands import tool
import json
import base64
from urllib.parse import urlparse
import os
import boto3
from typing import Any
from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import retrieve, image_reader
from strands.types.tools import ToolResult, ToolUse

s3 = boto3.client('s3', region_name=os.getenv("AWS_REGION"))
bedrock = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION"))
BUCKET_NAME = 'aw04-data'
IMAGE_FOLDER = 'images/'
FOLDER_PREFIX = 'looks/'
CLOUDFRONT_DOMAIN = 'https://d39bzdkvoca64w.cloudfront.net'

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-lite-v1:0",
)

def parse_filenames_from_string(filenames_str):
    s = filenames_str.strip().lstrip("[").rstrip("]")
    parts = s.split(",")
    urls = [p.strip().strip('"').strip("'") for p in parts if p.strip()]
    return urls

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
            "text": f"""
You are performing grounded visual analysis.

Multiple images of the same look may be provided.
Use all images collectively.
If a detail is visible in only one image, it counts as present.
If images conflict, prefer the clearest view.

STEP 1 — Visual Inventory
List all visible garments and accessories from top to bottom.
For each item include:
- Type
- Basic color (single word)
- Visible construction details
- Position on body
- Any visible hardware
If uncertain, state "unclear due to resolution."

Do not infer brand, season accuracy, or intent.

STEP 2 — Focused Scan
If the query involves:
- Jewelry: scan wrists, fingers, neck specifically.
- Layers: count neckline layers and sleeve layers separately.
- Closure: describe fastening mechanism before naming it.
- Lapels: describe shape before classifying.
- Hem: describe fold, stacking, or raw edge appearance.

STEP 3 — Answer
Answer the query using only confirmed observations.
If evidence is insufficient, state that clearly.

Query: {query}
"""
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
    
KB_PROMPT = """
Role:
Retrieve the look number, category, subcategory, primary and secondary color(s), pattern, primary and secondary outer material(s), and additional notes from the knowledge base based on the query.

Guidelines: 
If the look number is already included in the query, there is no need to retrieve it from the knowledge base.
Make sure the look number retrieved is a positive integer, and not a word or float. 
"""

VISUAL_PROMPT = """
Analyze look images for fit, silhouette, texture, and aesthetic details.

Guidelines:
Use the look number provided from the retrieved results to get the filenames using the get_look_images tool. 
Pass the retrieved image filenames into get_image_details in order to retrieve detailed visual analysis.
"""

SYNTHESIS_PROMPT = """
Role:
Synthesize a final answer based on visual and knowledge base information.

Guidelines:
Combine visual analysis with metadata for the final answer.
Report discrepancies between visual and metadata observations.
"""

@tool 
def get_kb_visual_analysis(query: str) -> str:
    kb_agent = Agent(model=bedrock_model,
        system_prompt=KB_PROMPT, tools=[retrieve])
    visual_agent = Agent(model=bedrock_model,
        system_prompt=VISUAL_PROMPT, tools=[get_look_images, get_image_details])
    synthesis_agent = Agent(model=bedrock_model,
        system_prompt=SYNTHESIS_PROMPT)

    kb_results = kb_agent(f"Retrieve the look number based on this query: {query}")
    visual_results = visual_agent(f"Based on the look number retrieved, answer the query. Retrived results: {kb_results}. Query: {query}.")
    response = synthesis_agent(f"Synthesize a final result for this query: {query}. Visual results: {visual_results}. Knowledge base results: {kb_results}.")
    return response

READER_PROMPT = """
Role:
Use the image_reader tool to format the image path from the query to be used in later steps.
Pass the image path from the query into the image_path parameter.

Guidelines:
Only format the image path and combine it with the query. 
Do not pass the entire query into image_reader, only the path.
If there is no image, indicate this and decline to answer. 
"""

IMAGE_KB_PROMPT = """
Retrieve any relevant images related to the image and query.

Guidelines:
If the image is irrelevant, indicate this and decline to answer.
If no results are able to be retrieved, state this.
"""

SUMMARIZER_PROMPT = """
Role:
Synthesize a final answer based on visual and knowledge base information.

Guidelines:
If the image is irrelevant, indicate this and decline to answer.
Combine visual analysis with metadata for the final answer.
"""

@tool 
def get_image_input(query: str) -> str:
    
    reader_agent = Agent(model=bedrock_model,
        system_prompt=KB_PROMPT, tools=[image_reader])
    visual_agent = Agent(model=bedrock_model,
        system_prompt=VISUAL_PROMPT, tools=[retrieve])
    synthesis_agent = Agent(model=bedrock_model,
        system_prompt=SYNTHESIS_PROMPT)

    reader_results = reader_agent(f"Format the image and query: {query}")
    
    kb_results = visual_agent.tool.retrieve(text=reader_results, knowledgeBaseId=os.getenv('IMAGE_KNOWLEDGE_BASE_ID'))
    
    # kb_results = visual_agent(f"Based on the formatted image and query, retrieve relevant results. Query: {reader_results}")
    response = synthesis_agent(f"Synthesize a final result for this query: {reader_results}. Knowledge base results: {kb_results}.")
    return response