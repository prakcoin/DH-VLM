import boto3
import json
import base64
import os
import re
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "amazon.nova-pro-v1:0"
IMAGE_DIR = "data/images"
OUTPUT_DIR = "data/labels"
IMAGE_REL_PATH = "data/images"
OUTPUT_FILE = "import_to_ls.json"

bedrock = boto3.client(service_name="bedrock-runtime", 
                       region_name="us-east-1",
                       aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                       aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_look_groups(directory):
    groups = defaultdict(list)
    
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            match = re.search(r'look(\d+)_(\d+)', filename)
            if match:
                look_id = int(match.group(1))
                seq_id = int(match.group(2))
                full_path = os.path.join(directory, filename)
                
                groups[look_id].append((seq_id, full_path))

    for look_id in groups:
        groups[look_id].sort()
        groups[look_id] = [path for _, path in groups[look_id]]

    return dict(sorted(groups.items()))

def analyze_look(look_id, image_paths):
    content = []
    for path in image_paths:
        content.append({
            "image": {
                "format": "jpeg",
                "source": {
                    "bytes": encode_image(path)
                }
            }
        })
    
    prompt = f"""
    Analyze the runway look in these images for {look_id}. 
    Identify every distinct piece of clothing and accessory worn by the model visible within the look images. 
    Return a single JSON list of objects. Each distinct piece of clothing or accessory worn by the model (e.g., the jacket, shirt, trousers, shoes, belt) must be its own separate object in the list.

    Each object must use these exact keys:
    "Name": (A concise, distinct name. Priority: [Material (Only include if it is a defining characteristic or unconventional for the item.)] [Misc Identifier] [Item Type]. Examples: "Leather Studded Belt", "Eye Graphic T-Shirt", "Rust Denim".),
    "Reference Code": "Not available",
    "Category": (Must be one of: Accessories, Bottom, Footwear, Outerwear, Top),
    "Subcategory": (e.g., T-Shirt, Belt, Jeans),
    "Primary Color": (Dominant color),
    "Secondary Color(s)": (List secondary colors or "No secondary color"),
    "Pattern": (e.g., Solid, Plaid, Striped),
    "Primary Outer Material": "None",
    "Secondary Outer Material(s)": "None",
    "Additional Notes": (Describe only distinguishing physical features like hardware, unique textures, specific fit, or distressing. If the item is a basic staple with no unique features, leave this as "Standard fit/design". Avoid subjective praise. Max 20 words.)
    """
    
    content.append({"text": prompt})

    body = json.dumps({
        "inferenceConfig": {
            "max_new_tokens": 2000,
            "temperature": 0.0
        },
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    })

    response = bedrock.invoke_model(
        modelId=MODEL_ID, 
        body=body
    )

    response_body = json.loads(response.get("body").read())
    try:
        return json.loads(response_body["output"]["message"]['content'][0]['text'])
    except Exception as e:
        print(f"Error parsing JSON for {look_id}:", e)
        return []
    
os.makedirs(OUTPUT_DIR, exist_ok=True)

groups = get_look_groups(IMAGE_DIR)
all_tasks = []

# Processing looks (limited to 3 for your testing as per original code)
for look_id, filenames in list(groups.items())[:3]:
    print(f"Processing Look {look_id}...")
    ai_items = analyze_look(look_id, filenames)
    
    # Create the Label Studio local storage paths
    ls_image_paths = [f"/data/local-files/?d={IMAGE_REL_PATH}/{f}" for f in filenames]

    for item in ai_items:
        # Map AI keys to your Label Studio XML keys
        data_dict = {
            "look_number": str(look_id),
            "name": item.get("Name", ""),
            "reference_code": item.get("Reference Code", ""),
            "category": item.get("Category", ""),
            "subcategory": item.get("Subcategory", ""),
            "primary_color": item.get("Primary Color", ""),
            "secondary_colors": item.get("Secondary Color(s)", ""),
            "pattern": item.get("Pattern", ""),
            "primary_outer_material": item.get("Primary Outer Material", ""),
            "secondary_outer_materials": item.get("Secondary Outer Material(s)", ""),
            "notes": item.get("Additional Notes", "")
        }

        # Fill image slots (image_0 to image_4)
        for i in range(5):
            data_dict[f"image_{i}"] = ls_image_paths[i] if i < len(ls_image_paths) else ""

        all_tasks.append({"data": data_dict})

# Save the final consolidated file
with open(OUTPUT_FILE, "w") as f:
    json.dump(all_tasks, f, indent=2)

print(f"Success! Created {len(all_tasks)} tasks in {OUTPUT_FILE}")