from strands import tool, Agent
from strands.models import BedrockModel
import boto3
import csv
import io
import os
import logging
import pandas as pd
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client('s3', region_name=os.getenv("AWS_REGION"))

BUCKET_NAME = 'aw04-data'
FOLDER_PREFIX = 'looks/'
CHUNK_SIZE = 9

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

def format_chunk(df_chunk: pd.DataFrame) -> str:
    lines = []
    for look_id in sorted(df_chunk['Look Number'].unique(), key=lambda x: int(x) if str(x).isdigit() else x):
        look_items = df_chunk[df_chunk['Look Number'] == look_id]
        lines.append(f"\nLook {look_id}:")
        for _, item in look_items.iterrows():
            line = f"  - {item['Name']} | {item['Subcategory']} | {item['Primary Color']}"
            if str(item.get('Pattern', '')).strip():
                line += f" | Pattern: {item['Pattern']}"
            if str(item.get('Primary Outer Material', '')).strip():
                line += f" | Material: {item['Primary Outer Material']}"
            if str(item.get('Secondary Outer Material(s)', '')).strip():
                line += f" / {item['Secondary Outer Material(s)']}"
            if str(item.get('Additional Notes', '')).strip():
                line += f" | Notes: {item['Additional Notes']}"
            lines.append(line)
    return "\n".join(lines)

def run_chunk_agent(chunk_text: str, query: str) -> str:
    model = BedrockModel(model_id="us.amazon.nova-2-lite-v1:0", temperature=0.0, max_tokens=4000)
    agent = Agent(
        model=model,
        system_prompt="""You are analyzing a subset of the Dior Homme Autumn/Winter 2004 "Victim of the Crime" collection.
Given collection data for a set of looks and a query, extract all information from your subset that is relevant to answering that query.
Match items by their description, material, category, and construction — not just their exact name. For example, a query about leather jackets should match any jacket whose name or material indicates it is made of leather, even if it is named differently (e.g. "Beetle Leather Riders Jacket", "Quilted Leather Riders Jacket").
Be specific — include item names, look numbers, and any relevant details. Only report what is present in your data. Do not infer or add information that is not explicitly present."""
    )
    response = agent(f"Collection data:\n{chunk_text}\n\nQuery: {query}\n\nExtract all relevant information from this subset.")
    return str(response)


def run_reduce_agent(partial_results: list, query: str) -> str:
    model = BedrockModel(model_id="us.amazon.nova-2-lite-v1:0", temperature=0.0, max_tokens=8000)
    agent = Agent(
        model=model,
        system_prompt="""You are synthesizing findings from multiple agents that each analyzed a subset of the Dior Homme Autumn/Winter 2004 "Victim of the Crime" collection.
Combine their findings into a single accurate and complete answer.
Rules:
- Only include information explicitly reported by the subset agents. Do not add, infer, or embellish.
- Deduplicate look numbers and item names — if the same look or item appears in multiple subsets, count it only once.
- Be specific with item names and look numbers."""
    )
    combined = "\n\n---\n\n".join([f"Subset {i+1} findings:\n{r}" for i, r in enumerate(partial_results)])
    response = agent(f"Query: {query}\n\nFindings from {len(partial_results)} subsets:\n\n{combined}\n\nProvide a complete synthesized answer.")
    return str(response)


@tool
def get_collection_inventory(
    query: str,
    subcategory: Optional[str] = None,
    color: Optional[str] = None
):
    """
    Perform aggregation and analysis over the full archive collection using a map-reduce approach.

    Use this for aggregation questions that retrieval cannot answer completely due to result limits,
    such as: counting how many looks contain a specific item type, identifying recurring motifs,
    listing all looks featuring a particular item, or surfacing item variations across the collection.

    The collection is split into subsets, each analyzed by a dedicated agent in parallel, with
    results synthesized into a complete answer.

    Args:
        query (str): The natural language question to answer about the collection.
        subcategory (str, optional): Pre-filter by subcategory to narrow the dataset before analysis (e.g. 'Knitwear').
        color (str, optional): Pre-filter by primary color to narrow the dataset before analysis (e.g. 'Black').

    Returns:
        A synthesized answer to the query drawn from the full collection.
    """
    df = df_archive.copy()

    if subcategory and subcategory.strip():
        sub_lower = subcategory.lower()
        df = df[df['Subcategory'].str.lower().apply(lambda v: v in sub_lower or sub_lower in v)]
    if color and color.strip():
        color_lower = color.lower()
        df = df[df['Primary Color'].str.lower().apply(lambda v: v in color_lower or color_lower in v)]

    if df.empty:
        return "No items match your query parameters."

    look_ids = sorted(df['Look Number'].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
    chunks = [look_ids[i:i + CHUNK_SIZE] for i in range(0, len(look_ids), CHUNK_SIZE)]

    logger.info(f"Running map-reduce over {len(look_ids)} looks in {len(chunks)} chunks")

    chunk_results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {
            executor.submit(
                run_chunk_agent,
                format_chunk(df[df['Look Number'].isin(chunk)]),
                query
            ): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                chunk_results[idx] = future.result()
                logger.info(f"Chunk {idx + 1}/{len(chunks)} complete")
            except Exception as e:
                logger.error(f"Chunk agent {idx + 1} failed: {e}")
                chunk_results[idx] = f"[Error processing subset {idx + 1}]"

    return run_reduce_agent(chunk_results, query)
