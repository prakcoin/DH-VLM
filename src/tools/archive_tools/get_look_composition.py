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
FOLDER_PREFIX = 'looks/'

@tool
def get_look_composition(look_number: str):
    """
    Retrieve archival composition data for a specific look.
    
    Use this tool when you want to list every item included in a specific runway look. Only use it when you have a look number.
    
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
