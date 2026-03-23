from strands import tool
import boto3
import csv
import io
import os
import logging
import pandas as pd
from typing import Optional

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
df_archive = pd.DataFrame(FULL_COLLECTION)

@tool
def get_collection_inventory(subcategory: Optional[str] = None, color: Optional[str] = None):
    """
    Perform a statistical and descriptive analysis of the archive.
    Provides total item counts, unique look distribution, and filtered inventory lists.

    Use this when you need to analyze recurring materials, patterns, or themes, 
    either for the whole collection or filtered by category or color.

    Args:
    subcategory (str, optional): Filter by subcategory (e.g., 'Knitwear'). 
    color (str, optional): Filter by primary color (e.g., 'Red').

    Returns:
    A dictionary containing subcategory distributions and a pipe-delimited inventory list formatted as Qty(UniqueLooks)|Name|Subcategory|Color|LookNumbers.
    """
    df = df_archive.copy()

    if subcategory and subcategory.strip():
        df = df[df['Subcategory'].str.lower() == subcategory.lower()]
        
    if color and color.strip():
        df = df[df['Primary Color'].str.lower() == color.lower()]

    if df.empty:
        return "No items match your query parameters."

    all_look_ids = set()
    for val in df['Look Number'].dropna().astype(str):
        for part in val.split(','):
            clean_look = part.strip()
            if clean_look:
                all_look_ids.add(clean_look)

    summary = df.groupby(['Name', 'Subcategory', 'Primary Color']).agg(
        total_items=('Look Number', 'count'),
        unique_look_count=('Look Number', 'nunique'),
        look_list=('Look Number', lambda x: ", ".join(sorted(x.unique().astype(str))))
    ).reset_index()

    report = ["Qty(UniqueLooks)|Name|Subcategory|Color|LookNumbers"]
    for _, row in summary.iterrows():
        qty_label = f"{row['total_items']}({row['unique_look_count']}L)" if row['total_items'] != row['unique_look_count'] else str(row['total_items'])
        report.append(f"{qty_label}|{row['Name']}|{row['Subcategory']}|{row['Primary Color']}|{row['look_list']}")

    return {
        "query_metadata": {
            "total_items_found": len(df),
            "total_unique_looks": len(all_look_ids),
            "subcategory_distribution": df['Subcategory'].value_counts().to_dict()
        },
        "inventory_data": report
    }