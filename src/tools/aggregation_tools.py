from strands import tool
import boto3
import os
import csv
import io
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="us-east-1"
)

BUCKET_NAME = 'aw04-data'
FOLDER_PREFIX = 'looks/'

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

@tool
def get_collection_items(search_query: str) -> str: 
    """
    Retrieve all items across the collection that match a multi-term search query.

    Use this tool when you need to filter the archive by materials, categories, item types, or descriptive terms and return a list of matching items across multiple looks.

    Args:
    search_query (str): Keywords describing the items to retrieve (e.g., "leather jacket", "wool coat black").

    Returns:
    Textual response listing matching items formatted as: Item Name (Look X)
    If no matches are found, returns a message indicating no results.
    """
    terms = str(search_query).lower().strip().split() 

    if not terms: 
        return "Please provide a search query." 
    
    matches = [] 
    all_items = load_full_collection() 
    
    for item in all_items: 
        name = item.get('Name', '') 
        notes = item.get('Additional Notes', '') 
        subcat = item.get('Subcategory', '') 
        primary_mat = item.get('Primary Outer Material', '') 
        secondary_mat = item.get('Secondary Outer Material(s)', '') 
        look_id = item.get('Look Number', '') 
        
        row_text = f"Name: {name} Category: {subcat} Primary Outer Material: {primary_mat} Secondary Outer Material(s): {secondary_mat} Notes: {notes}".lower() 
        
        if all(t in row_text for t in terms): 
            row_output = f"{name.capitalize()} (Look {look_id})" 
            matches.append(row_output) 
            
    if not matches: 
        return f"I couldn't find any items matching '{search_query}' in the collection." 
    
    return f"Archival inventory for '{search_query}': {', '.join(matches)}"

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
    items = load_full_collection()
    
    unique_items_map = {}
    
    for item in items:
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
    for sig, data in unique_items_map.items():
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
    items = load_full_collection()
    terms = str(search_query).lower().strip().split()
    
    if not terms:
        return "0"

    unique_looks = set()
    
    for item in items:
        text = f"{item.get('Name','')} {item.get('Additional Notes','')} {item.get('Subcategory','')} {item.get('Primary Outer Material','')} {item.get('Secondary Outer Material(s)','')}".lower()
        
        if all(t in text for t in terms):
            unique_looks.add(item.get('Look Number'))

    return str(len(unique_looks))

