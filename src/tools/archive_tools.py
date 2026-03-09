from strands import tool
import boto3
import csv
import io
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client('s3', region_name=os.getenv("AWS_REGION"))

IMAGE_FOLDER = 'images/'
BUCKET_NAME = 'aw04-data'
FOLDER_PREFIX = 'looks/'
CLOUDFRONT_DOMAIN = 'https://d39bzdkvoca64w.cloudfront.net'

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

def load_full_collection():
    all_items = []
    objects = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=FOLDER_PREFIX)
    for obj in objects.get('Contents', []):
        if not obj['Key'].endswith('.csv'): continue
        try:
            content = s3.get_object(Bucket=BUCKET_NAME, Key=obj['Key'])
            look_id = obj['Key'].split('_')[-1].replace('.csv', '')
            reader = csv.DictReader(io.StringIO(content['Body'].read().decode('utf-8-sig')))
            for row in reader:
                row['Look Number'] = look_id
                all_items.append(row)
        except Exception as e:
            logger.error(f"Error reading {obj['Key']}: {str(e)}")
    return all_items

FULL_COLLECTION = load_full_collection()

@tool
def get_collection_summary():
    """
    Generate a summary of the entire collection for motif or pattern analysis.

    Use this tool when you need a consolidated inventory overview to analyze recurring materials, silhouettes, patterns, or design themes across the archive.

    Args:
    None

    Returns:
    A formatted textual inventory summary.
    """
    unique_items_map = {}
    
    for item in FULL_COLLECTION:
        signature = (
            item.get('Name', ''),
            item.get('Subcategory', ''),
            item.get('Primary Outer Material', ''),
            item.get('Secondary Outer Material(s)', ''),
            item.get('Additional Notes', ''),
            item.get('Pattern', '')
        )
        
        if signature not in unique_items_map:
            unique_items_map[signature] = item

    unique_list = []
    for _, data in unique_items_map.items():
        desc = (
            f"- {data.get('Name')}: {data.get('Subcategory')} in {data.get('Primary Outer Material')} and {data.get('Secondary Outer Material(s)')}. "
            f"Pattern: {data.get('Pattern')}. Notes: {data.get('Additional Notes')}"
        )
        unique_list.append(desc)

    inventory_string = "\n".join(unique_list)

    return (
        "Analyze the following inventory to identify recurring motifs. \n\n"
        f"Unique Inventory:\n{inventory_string}"
    )

@tool
def get_item_counts(search_query: str) -> str:
    """
    Count how many unique looks contain items matching a given search query.

    Use this tool when the question requires quantitative aggregation, such as determining how frequently a material, garment type, or motif appears across the collection. 

    Args:
    search_query (str): Keywords describing the attribute or item type to count (e.g., "leather", "striped shirt").

    Returns:
    A string representing the number of unique looks that contain matching items.
    Returns "0" if no matches are found.
    """
    terms = str(search_query).lower().strip().split()
    
    if not terms:
        return "0"

    unique_looks = set()
    
    for item in FULL_COLLECTION:
        text = f"{item.get('Name','')} {item.get('Additional Notes','')} {item.get('Subcategory','')} {item.get('Primary Outer Material','')} {item.get('Secondary Outer Material(s)','')}".lower()
        
        if all(t in text for t in terms):
            unique_looks.add(item.get('Look Number'))

    return str(len(unique_looks))
