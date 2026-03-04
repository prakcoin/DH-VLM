from strands import tool
import boto3
import csv
import io
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client('s3', region_name="us-east-1")

IMAGE_FOLDER = 'images/'
BUCKET_NAME = 'aw04-data'
FOLDER_PREFIX = 'looks/'
CLOUDFRONT_DOMAIN = "https://d39bzdkvoca64w.cloudfront.net"

@tool
def get_look_composition(look_number: str):
    """
    Retrieve archival composition data for a specific look.
    
    Use this tool when you want to list every item included in a specific runway look.
    
    Args:
    look_number (str): The unique identifier for the look, e.g., "1".

    Returns: 
    A list of every item in the requested look.
    """
    clean_id = str(look_number).strip().lower().replace('look', '').strip()
    target_file = f"{FOLDER_PREFIX}look_{clean_id}.csv"
    
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=target_file)
        content = response['Body'].read().decode('utf-8-sig')
        data = list(csv.DictReader(io.StringIO(content)))
        return f"Archival data for Look {look_number}: {str(data)}"
    except s3.exceptions.NoSuchKey:
        logger.error(f"File not found: {target_file}")
        return f"I'm sorry, I couldn't find archival data for Look {look_number}."
    except Exception as e:
        logger.error(f"Error fetching {target_file}: {str(e)}")
        return f"I'm sorry, I couldn't find archival data for Look {look_number}."

@tool
def get_look_images(look_number: str):
    """
    Retrieve the runway images for a specific look.
    
    Use this tool when a user asks to see a specific runway look.
    
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
def get_item_attribute(item_name: str, look_number: str = None, attribute_type: str = None):
    """
    Search archival data for items matching a name and optionally a look number or attribute type.
    
    Use this to find specific details (like reference codes, materials, or design features) for a specific named item. 
    
    Args:
    item_name (str): Name or description of the item to search.
    look_number (str, optional): The unique identifier for the look, e.g., "1".
    attribute_type (str, optional): Specify which attribute you want extracted.

    Returns: 
    A ranked list of matching items with all available metadata.
    """
    search_terms = str(item_name).lower().strip().split()
    
    scored_results = []
    
    objects = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=FOLDER_PREFIX)
    for obj in objects.get('Contents', []):
        if not obj['Key'].endswith('.csv'): continue
        if look_number and f"look_{look_number}.csv" not in obj['Key']: continue
            
        content = s3.get_object(Bucket=BUCKET_NAME, Key=obj['Key'])
        reader = csv.DictReader(io.StringIO(content['Body'].read().decode('utf-8-sig')))
        
        for row in reader:
            row_text = f"{row.get('Name','')} {row.get('Additional Notes','')} {row.get('Primary Outer Material','')} {row.get('Secondary Outer Material(s)','')}".lower()
            
            match_count = sum(1 for term in search_terms if term in row_text)
            
            if match_count > 0:
                scored_results.append({
                    'score': match_count,
                    'data': row
                })

    if not scored_results:
        return f"No archival matches found for '{item_name}'."

    scored_results.sort(key=lambda x: x['score'], reverse=True)
    
    best_matches = [item['data'] for item in scored_results[:10]]

    return f"Archival search results (ranked by relevance): {str(best_matches)}. Extract the {attribute_type} for the {item_name}."
