from strands import tool
import boto3
import csv
import io
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client('s3', region_name=os.getenv("AWS_REGION"))

BUCKET_NAME = 'aw04-data'
IMAGE_FOLDER = 'images/'
FOLDER_PREFIX = 'looks/'
CLOUDFRONT_DOMAIN = 'https://d39bzdkvoca64w.cloudfront.net'

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