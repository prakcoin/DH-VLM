from strands import Agent, tool
from strands.models import BedrockModel
import boto3
import csv
import io
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="us-east-1"
)

IMAGE_FOLDER = 'images/'
BUCKET_NAME = 'aw04-data' 
FOLDER_PREFIX = 'looks/'
CLOUDFRONT_DOMAIN = "https://d39bzdkvoca64w.cloudfront.net"

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-lite-v1:0",
)

@tool
def get_look_composition(look_number: str):
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

ITEM_PROMPT = """
Role:
Handle questions about individual items, looks, and metadata.

Guidelines:
Never ask the user for look numbers; retrieve directly from the database.
Pass image filenames to VisualAgent if detailed visual analysis is requested.
"""

@tool
def item_assistant(query: str) -> str:
    try:
        item_agent = Agent(
            model=bedrock_model,
            system_prompt=ITEM_PROMPT,
            tools=[get_look_images, get_look_composition, get_item_attribute]
        )

        response = item_agent(query)
        return str(response)
    except Exception as e:
        return f"Error in item assistant: {str(e)}"