from strands import tool
import boto3
import csv
import io
import os
import logging
from collections import Counter

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client('s3', region_name=os.getenv("AWS_REGION"))

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

FULL_COLLECTION = load_full_collection()

@tool
def get_collection_summary():
    """
    Generate a statistical and descriptive summary of the entire collection,
    including item frequency and subcategory distribution.

    Use this tool when you need a consolidated inventory overview to analyze recurring materials, silhouettes, patterns, or design themes across the archive.

    Args:
    None

    Returns:
    A report containing subcategory counts and a frequency-weighted inventory.
    """
    unique_items_map = {}
    item_frequency = Counter()
    subcategory_counts = Counter()
    
    for item in FULL_COLLECTION:
        signature = (
            item.get('Name', ''),
            item.get('Look Number', ''),
            item.get('Primary Color', ''),
        )
        
        item_frequency[signature] += 1
        subcat = item.get('Subcategory', 'Uncategorized')
        subcategory_counts[subcat] += 1
        
        if signature not in unique_items_map:
            unique_items_map[signature] = item

    stats_header = "Inventory Statistics\n"
    stats_header += f"Total Unique Items: {len(unique_items_map)}\n"
    stats_header += "Subcategory Distribution:\n"
    for subcat, count in subcategory_counts.most_common():
        stats_header += f"- {subcat}: {count}\n"

    unique_list = []
    sorted_signatures = item_frequency.most_common()

    for signature, count in sorted_signatures:
        data = unique_items_map[signature]
        desc = (
            f"- {data.get('Name')} ({count} items): {data.get('Subcategory')} | {data.get('Primary Color')}"
        )
        unique_list.append(desc)

    inventory_string = "\n".join(unique_list)

    return (
        f"{stats_header}\n"
        "Full Inventory (Sorted by Frequency):\n"
        f"{inventory_string}"
    )
