from strands import tool
import os
import boto3
from strands.models import BedrockModel

s3 = boto3.client('s3', region_name=os.getenv("AWS_REGION"))
bedrock = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION"))
BUCKET_NAME = 'aw04-data'
IMAGE_FOLDER = 'images/'
FOLDER_PREFIX = 'looks/'
CLOUDFRONT_DOMAIN = 'https://d39bzdkvoca64w.cloudfront.net'

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-2-lite-v1:0",
)

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