from strands import tool
import json
import base64
from urllib.parse import urlparse
import os
import boto3
from strands import Agent, tool
from strands_tools import stop
from strands.models import BedrockModel
from src.agents.hooks import LimitToolCounts
from botocore.config import Config as BotocoreConfig

s3 = boto3.client('s3', region_name=os.getenv("AWS_REGION"))
bedrock = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION"))

BUCKET_NAME = 'aw04-data'
IMAGE_FOLDER = 'images/'
CLOUDFRONT_DOMAIN = 'https://d39bzdkvoca64w.cloudfront.net'

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-2-lite-v1:0",
)

COMPARISON_PROMPT = """
Role:
Analyze IMAGE A (Query) and IMAGE B (Retrieved) to verify visual relevance.

Guidelines:
Determine if these two images are visually related or if this retrieval is a false positive.
Categorize the connection as one of the following: Direct match, Strong relation, Weak relation, No relation
Output a short analysis of each image
Provide a one-sentence summary of the relationship.
"""

IMAGE_KB_PROMPT = """ 
Role:
Retrieve the most relevant images related to the image using the image_retrieve and get_cloudfront_url tools.

Guidelines:
Pass the image path (PNG, JPEG/JPG, GIF, or WebP formats) from the query into the image_path parameter of the image_retrieve tool.
For every file path or filename returned by image_retrieve, you MUST call the get_cloudfront_url tool to generate a valid access link.
If no image is found, or no image path is provided, use the stop tool with reason IMAGE_NOT_AVAILABLE.
If retrieve returns no results or an error, use the stop tool with reason INFO_NOT_AVAILABLE.
Make sure the full image path is passed, and not just the filename.
Return the full filepaths for the matching images if they are found.
If the top result scores end in a tie, return all of the results. 
"""

IMAGE_READER_PROMPT = """ 
Role:
Analyze the query image and retrieved images using the get_image_comparison tool.

Guidelines: 
For each image analysis, pass the query image path (PNG, JPEG/JPG, GIF, or WebP formats) into the query_filename parameter and each image one at a time in the retrieved_filename parameter.
If the query image path is missing, or if the retrieval results contain no valid image paths, use the stop tool with reason IMAGE_NOT_AVAILABLE.
There must be at least two images analyzed: the query image and at least one retrieved image.
If the images compared are completely unrelated, state this.
"""

SYNTHESIS_PROMPT = """
Role:
Synthesize a final answer based on visual and knowledge base information.
If the results aren't conclusive, state this.

Guidelines:
Combine visual analysis with metadata for the final answer.
Report discrepancies between visual and metadata observations.
"""

@tool
def image_retrieve(image_path: str) -> str:
    """
    Perform image-retrieval from the image knowledge base.

    Use this tool when a query requires direct visual inspection of garments, accessories, layering, closures, construction details, or physical attributes that cannot be reliably inferred from metadata alone. 

    Args:
    image_path (str): The image to use as a retrieval key.

    Returns:
    A structured textual analysis based only on confirmed visual observations.
    """
    kb_id = os.getenv("IMAGE_KNOWLEDGE_BASE_ID")
    region_name = os.getenv("AWS_REGION")
    min_score = 0.4
    
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = base64.b64encode(image_file.read()).decode("utf-8")

        config = BotocoreConfig(user_agent_extra="archival-research-agent")
        client = boto3.client("bedrock-agent-runtime", region_name=region_name, config=config)

        image_format = image_path.split('.')[-1].lower()
        if image_format == 'jpg':
            image_format = 'jpeg'

        retrieval_query = {
            "type": "IMAGE",
            "image": {
                "format": image_format,
                "inlineContent": base64.b64decode(image_bytes)
            }
        }

        retrieval_config = {
            "vectorSearchConfiguration": {
                "numberOfResults": 3
            }
        }

        response = client.retrieve(
            retrievalQuery=retrieval_query,
            knowledgeBaseId=kb_id,
            retrievalConfiguration=retrieval_config
        )

        all_results = response.get("retrievalResults", [])
        
        filtered_results = [r for r in all_results if r.get("score", 0.0) >= min_score]
        
        if not filtered_results:
            return f"No results found above score threshold of {min_score}."

        formatted = []
        for result in filtered_results:
            location = result.get("location", {})
            
            doc_id = "Unknown"
            if "s3Location" in location:
                doc_id = location["s3Location"].get("uri", "Unknown")
            elif "customDocumentLocation" in location:
                doc_id = location["customDocumentLocation"].get("id", "Unknown")
            
            score = result.get("score", 0.0)
            formatted.append(f"\nScore: {score:.4f}")
            formatted.append(f"Document ID: {doc_id}")

            content = result.get("content", {})
            if content and isinstance(content.get("text"), str):
                formatted.append(f"Content: {content['text']}\n")

        formatted_results_str = "\n".join(formatted)

        return f"Retrieved {len(filtered_results)} results with score >= {min_score}:\n{formatted_results_str}"
    except Exception as e:
        return f"Error during visual retrieval: {str(e)}"

@tool
def get_cloudfront_url(image_filename: str):
    """
    Convert a single archival filename into its corresponding CloudFront URL.
    
    Args:
    image_filename (str): The specific filename.

    Returns: 
    str: The full CloudFront URL for that specific image.
    """
    clean_name = image_filename.split('/')[-1].replace('`', '').strip()
    full_url = f"{CLOUDFRONT_DOMAIN}/{IMAGE_FOLDER}{clean_name}"
    
    return full_url

@tool
def get_image_comparison(query_filename: str, retrieved_filename: str):
    """
    Perform a direct side-by-side visual comparison between a query image and a single archival look image.

    Args:
    query_filename (str): Local path to the user's query image.
    retrieved_filename (str): The filename or URL of the archival image.

    Returns:
    A analysis and comparison of the images.
    """
    try:
        content_blocks = []

        with open(query_filename, "rb") as f:
            query_bytes = f.read()
        
        content_blocks.append({"text": "IMAGE A (Query):"})
        content_blocks.append({
            "image": {
                "format": "jpeg",
                "source": {"bytes": base64.b64encode(query_bytes).decode("utf-8")}
            }
        })

        clean_retrieved = os.path.basename(urlparse(retrieved_filename).path)
        image_key = f"{IMAGE_FOLDER}{clean_retrieved}"

        response = s3.get_object(Bucket=BUCKET_NAME, Key=image_key)
        retrieved_bytes = response['Body'].read()

        content_blocks.append({"text": "IMAGE B (Retrieved):"})
        content_blocks.append({
            "image": {
                "format": "jpeg",
                "source": {"bytes": base64.b64encode(retrieved_bytes).decode("utf-8")}
            }
        })

        content_blocks.append({
            "text": COMPARISON_PROMPT
        })

        body = json.dumps({
            "inferenceConfig": {
                "max_new_tokens": 800,
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
        return f"Error comparing {query_filename} and {retrieved_filename}: {str(e)}"

@tool 
def get_image_input(query: str) -> str:
    """
    Perform a direct side-by-side visual comparison between a query image and a single archival look image.

    Args:
    query (str): A query that includes an image.

    Returns:
    An answer to the query and image.
    """
    limit_hook = LimitToolCounts(max_tool_counts={"image_retrieve": 3})

    retrieval_agent = Agent(model=bedrock_model,
        system_prompt=IMAGE_KB_PROMPT, tools=[image_retrieve, get_cloudfront_url, stop], hooks=[limit_hook])
    
    visual_agent = Agent(model=bedrock_model,
        system_prompt=IMAGE_READER_PROMPT, tools=[get_image_comparison, stop])
    
    synthesis_agent = Agent(model=bedrock_model,
        system_prompt=SYNTHESIS_PROMPT)

    kb_results = retrieval_agent(f"From the image in the query, retrieve the best match image(s). "
                                 f"Query: {query}.")
    if not str(kb_results).strip():
        return "No matching image(s) found in the knowledge base."
    validation_results = visual_agent(
        f"Compare the user's image in the query with these archival images and URLs. " 
        f"Query: {query}"
        f"Knowledge base metadata: {str(kb_results)}. "
    )
    if not str(validation_results).strip():
        return "Visual comparison is currently unavailable."
    response = synthesis_agent(f"Synthesize a final result for this query. "
                               f"Query: {query}"
                               f"Knowledge base metadata: {str(kb_results)}. "
                               f"Visual validation analysis: {str(validation_results)}.")
    return response