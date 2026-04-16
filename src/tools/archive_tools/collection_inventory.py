from strands import tool, Agent
from strands.models import BedrockModel
from src.agents.hooks import LimitToolCounts
import boto3
import csv
import io
import os
import logging
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client('s3', region_name=os.getenv("AWS_REGION"))

BUCKET_NAME = 'aw04-data'
FOLDER_PREFIX = 'looks/'

SCHEMA_CONTEXT = """
DataFrame variable: df
Columns:
- Name (str): item name, e.g. 'Striped T-Shirt', 'Suede Moto Boot', 'V-Neck Knit'
- Reference Code (str): Dior reference; 'Not available' or 'To be updated' when unknown
- Look Number (int): 1-45
- Category (str): 'Accessories', 'Top', 'Bottom', 'Footwear', 'Outerwear'
- Subcategory (str): 'T-Shirt', 'Belt', 'Boot', 'Bracelet', 'Knitwear', 'Trousers', 'Jeans', 'Overcoat', 'Scarf', 'Sneaker', 'Shirt', 'Blazer', 'Handkerchief', 'Eyewear', 'Gloves', 'Skirt', 'Riders Jacket', 'Tie', 'Flight Jacket', 'Necklace', 'Trucker Jacket', 'Peacoat', 'Bomber Jacket', 'Vest', 'Bag', 'Varsity Jacket', 'Coat', 'Zip Knitwear', 'Brooch', 'Ring'
- Primary Color (str): e.g. 'Black', 'White', 'Beige', 'Grey', 'Dark Grey', 'Light Grey', 'Red', 'Mustard', 'Burgundy', 'Tan', 'Indigo', 'Brown', 'Silver'
- Secondary Color(s) (str): 'No secondary color' when absent    
- Pattern (str): 'Solid', 'Striped', 'Washed', 'Speckled', 'Herringbone', 'Ribbed', 'Tartan', 'Gradient', 'Houndstooth'
- Primary Outer Material (str): e.g. 'Leather', 'Laine wool', 'Cotton', 'Suede', 'Polyester', 'Brass', 'Acetate'
- Secondary Outer Material(s) (str): 'None' when absent
- Additional Notes (str): free text describing construction details and off-runway variants

Aggregation patterns by query type:
- "how many looks contain X"       → df[<filter>]['Look Number'].nunique()
- "which looks contain X"          → sorted(df[<filter>]['Look Number'].unique().tolist())
- "how many total X items"         → len(df[<filter>])
- "how many distinct item types"   → df['Subcategory'].nunique()
- "list all distinct X"            → sorted(df[<filter>]['Name'].unique().tolist())
- "most common X"                  → df['<col>'].value_counts().idxmax()
- "count by X"                     → df.groupby('<col>').size().sort_values(ascending=False).to_dict()
- "recurring motifs / themes"      → call execute_pandas_expression twice: once for recurring items (df.groupby('Name')['Look Number'].nunique().sort_values(ascending=False).to_dict()) and once for recurring features (df[df['Additional Notes'].str.contains('<feature>', case=False, na=False)][['Name', 'Look Number', 'Additional Notes']].to_dict(orient='records')), then synthesize both
- "items not on runway / variants" → df[df['Additional Notes'].str.contains('not featured|additionally', case=False, na=False)][['Name', 'Look Number', 'Additional Notes']].to_dict(orient='records')

Matching guidelines:
- For semantic matches (e.g. "leather jacket"), combine filters across Name, Subcategory, and Primary Outer Material using | (OR)
- Use str.contains(..., case=False, na=False) for all string matching
- For "how many looks" questions always use ['Look Number'].nunique() — never len() — because a single look can contain multiple matching items (e.g. two scarves in one look), so len() overcounts looks
- For "what styles/types appear" queries, return distinct Names — not Subcategory values — as Names are the specific item identifiers (e.g. 'Suede Moto Boot', 'B01 Sneaker', not 'Boot', 'Sneaker')
- For off-runway variants, search Additional Notes for keywords like 'not featured', 'not on the runway', 'additionally'
"""

EVAL_NAMESPACE = {
    "__builtins__": {},
    "df": None,  # set at runtime
    "pd": pd,
    "sorted": sorted,
    "len": len,
    "list": list,
    "str": str,
    "int": int,
}


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
                row['Look Number'] = int(look_id)
                all_items.append(row)
        except Exception as e:
            logger.error(f"Error reading {obj['Key']}: {str(e)}")
    return all_items

FULL_COLLECTION = load_full_collection()
df_archive = pd.DataFrame(FULL_COLLECTION)


def result_to_string(result) -> str:
    if isinstance(result, pd.DataFrame):
        return result.to_string()
    if isinstance(result, pd.Series):
        return result.to_string()
    return str(result)


@tool
def execute_pandas_expression(expression: str) -> str:
    """
    Execute a pandas expression against the full collection DataFrame and return the result as a string.

    The DataFrame is available as `df`. Use this to answer any counting, filtering, or
    aggregation question about the collection deterministically.

    Args:
        expression (str): A valid Python/pandas expression that operates on `df`.

    Returns:
        The string representation of the result.
    """
    namespace = {**EVAL_NAMESPACE, "df": df_archive}
    try:
        result = eval(expression, namespace)
        result_str = result_to_string(result)
        logger.info(f"Executed: {expression} → {result_str[:200]}")
        return result_str
    except Exception as e:
        logger.error(f"Expression failed: {expression!r} — {e}")
        return f"Error executing expression: {e}"


@tool
def get_collection_inventory(query: str) -> str:
    """
    Answer aggregation and counting questions about the full archive collection.

    Use this for questions that require looking across the entire collection, such as:
    counting how many looks or items match a type, finding the most common color or
    material, listing distinct item types, identifying recurring motifs, or surfacing
    off-runway item variations. Supports look-level, item-level, and type-level aggregation.

    The agent generates a pandas expression from the query and executes it via the
    execute_pandas_expression tool, guaranteeing accurate counts at any aggregation level.

    Args:
        query (str): The natural language question to answer about the collection.

    Returns:
        A synthesized answer to the query drawn from the full collection.
    """
    model = BedrockModel(model_id="us.amazon.nova-2-lite-v1:0", temperature=0.0, max_tokens=4000)
    limit_hook = LimitToolCounts(max_tool_counts={"execute_pandas_expression": 5})
    agent = Agent(
        model=model,
        system_prompt=f"""You answer questions about the Dior Homme Autumn/Winter 2004 "Victim of the Crime" collection.

You have access to the execute_pandas_expression tool, which runs a pandas expression against the full collection DataFrame and returns the exact result.

{SCHEMA_CONTEXT}

To answer a question:
1. Determine the correct pandas expression based on the query type and matching guidelines above
2. Call execute_pandas_expression with that expression
3. Use the exact result to provide a complete, accurate answer — do not recount or second-guess the numbers
4. For motif or theme questions, name the specific recurring items (e.g. 'Suede Moto Boot', 'Bandana Bracelet') — do not abstract them into generic category labels like 'belts' or 'leather items'. Also consider recurring design features (e.g. leather elbow patches, whiskering, striped patterns) found in Additional Notes and Pattern, not just recurring item names""",
        tools=[execute_pandas_expression],
        hooks=[limit_hook],
        callback_handler=None
    )
    return str(agent(query))
