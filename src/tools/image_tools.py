from strands import tool
import json
import base64
from urllib.parse import urlparse
import os
import boto3

s3 = boto3.client('s3', region_name="us-east-1")
bedrock = boto3.client('bedrock-runtime')
BUCKET_NAME = 'aw04-data'

def parse_filenames_from_string(filenames_str):
    s = filenames_str.strip().lstrip("[").rstrip("]")
    parts = s.split(",")
    urls = [p.strip().strip('"').strip("'") for p in parts if p.strip()]
    return urls

@tool
def get_image_details(image_filenames, query: str):
    """
    Perform grounded visual analysis on one or more look images.

    Use this tool when a query requires direct visual inspection of garments, accessories, layering, closures, construction details, or physical attributes that cannot be reliably inferred from metadata alone. 

    Args:
    image_filenames (list): One or more image filenames or URLs associated with a look.
    query (str): A specific visual question to answer.

    Returns:
    A structured textual analysis based only on confirmed visual observations.
    """
    try:
        if not image_filenames:
            return "Error: No image filenames provided."

        if isinstance(image_filenames, str):
            image_filenames = parse_filenames_from_string(image_filenames)

        content_blocks = []

        for filename in image_filenames:
            parsed = urlparse(filename)
            clean_filename = os.path.basename(parsed.path)
            image_key = f"images/{clean_filename}"

            response = s3.get_object(Bucket=BUCKET_NAME, Key=image_key)
            image_bytes = response['Body'].read()

            content_blocks.append({
                "image": {
                    "format": "jpeg",
                    "source": {
                        "bytes": base64.b64encode(image_bytes).decode("utf-8")
                    }
                }
            })

        content_blocks.append({
            "text": f"""
You are performing grounded visual analysis.

Multiple images of the same look may be provided.
Use all images collectively.
If a detail is visible in only one image, it counts as present.
If images conflict, prefer the clearest view.

STEP 1 — Visual Inventory
List all visible garments and accessories from top to bottom.
For each item include:
- Type
- Basic color (single word)
- Visible construction details
- Position on body
- Any visible hardware
If uncertain, state "unclear due to resolution."

Do not infer brand, season accuracy, or intent.

STEP 2 — Focused Scan
If the query involves:
- Jewelry: scan wrists, fingers, neck specifically.
- Layers: count neckline layers and sleeve layers separately.
- Closure: describe fastening mechanism before naming it.
- Lapels: describe shape before classifying.
- Hem: describe fold, stacking, or raw edge appearance.

STEP 3 — Answer
Answer the query using only confirmed observations.
If evidence is insufficient, state that clearly.

Query: {query}
"""
        })

        body = json.dumps({
            "inferenceConfig": {
                "max_new_tokens": 700,
                "temperature": 0.0
            },
            "messages": [
                {
                    "role": "user",
                    "content": content_blocks
                }
            ]
        })

        response = bedrock.invoke_model(
            modelId="amazon.nova-pro-v1:0",
            body=body
        )

        response_body = json.loads(response.get("body").read())
        return response_body["output"]["message"]["content"][0]["text"]

    except Exception as e:
        return f"Error analyzing images {image_filenames}: {str(e)}"