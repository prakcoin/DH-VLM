import boto3
import json
import base64
import os
import re
import pandas as pd
from label_studio_sdk import LabelStudio
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "amazon.nova-pro-v1:0"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "..", "data", "images")

bedrock = boto3.client(service_name="bedrock-runtime", 
                       region_name="us-east-1",
                       aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                       aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),)

ls = LabelStudio(base_url=f'http://{os.getenv("LABEL_STUDIO_URL")}:{os.getenv("LABEL_STUDIO_PORT")}', api_key=os.getenv("LABEL_STUDIO_API_KEY"))

"""
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/prak/DH-VLM/
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
"""

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

groups = get_look_groups(IMAGE_DIR)
max_images = max(len(filenames) for filenames in groups.values())

image_tags = "\n".join([
    f'<Image name="img{i}" value="$image_{i}" maxWidth="100%"/>' 
    for i in range(max_images)
])

label_config = f"""
<View name="root" style="display: flex; flex-direction: row; justify-content: space-between; height: 100vh; overflow: hidden;">

  <View name="image_container" style="flex: 1; padding: 10px; overflow-y: auto; border-right: 1px solid #ccc;">
    <Header name="h_look" value="Look $look_number"/>
    {image_tags}
  </View>

  <View name="form_container" style="flex: 0 0 350px; padding: 20px; overflow-y: auto; background: #f9f9f9;">
    <Header name="h_info" value="Garment Info" style="margin-top: 0;"/>

    <View name="v_name_group" style="margin-bottom: 15px;">
      <Text name="t_name" value="Name (Material, Identifier, Item Type)"/>
      <TextArea name="name" toName="img0" value="$name" rows="1" editable="true"/>
    </View>

    <View name="v_ref_group" style="margin-bottom: 15px;">
      <Text name="t_ref" value="Reference Code"/>
      <TextArea name="reference_code" toName="img0" value="$reference_code" rows="1" editable="true"/>
    </View>

    <View name="v_cat_group" style="margin-bottom: 15px;">
      <Text name="t_cat" value="Category (Accessories, Bottom, Footwear, Outerwear, Top)"/>
      <TextArea name="category" toName="img0" value="$category" rows="1" editable="true"/>
    </View>

    <View name="v_subcat_group" style="margin-bottom: 15px;">
      <Text name="t_subcat" value="Subcategory"/>
      <TextArea name="subcategory" toName="img0" value="$subcategory" rows="1" editable="true"/>
    </View>

    <View name="v_pcol_group" style="margin-bottom: 15px;">
      <Text name="t_pcol" value="Primary Color"/>
      <TextArea name="primary_color" toName="img0" value="$primary_color" rows="1" editable="true"/>
    </View>

    <View name="v_scol_group" style="margin-bottom: 15px;">
      <Text name="t_scol" value="Secondary Color(s)"/>
      <TextArea name="secondary_colors" toName="img0" value="$secondary_colors" rows="1" editable="true"/>
    </View>

    <View name="v_patt_group" style="margin-bottom: 15px;">
      <Text name="t_patt" value="Pattern"/>
      <TextArea name="pattern" toName="img0" value="$pattern" rows="1" editable="true"/>
    </View>

    <View name="v_pmat_group" style="margin-bottom: 15px;">
      <Text name="t_pmat" value="Primary Outer Material"/>
      <TextArea name="primary_outer_material" toName="img0" value="$primary_outer_material" rows="1" editable="true"/>
    </View>

    <View name="v_smat_group" style="margin-bottom: 15px;">
      <Text name="t_smat" value="Secondary Outer Material(s)"/>
      <TextArea name="secondary_outer_materials" toName="img0" value="$secondary_outer_materials" rows="1" editable="true"/>
    </View>

    <View name="v_notes_group" style="margin-bottom: 15px;">
      <Text name="t_notes" value="Additional Notes (Standard baseline, Features list, Alternative ref codes, Additional context)"/>
      <TextArea name="notes" toName="img0" value="$notes" rows="3" editable="true"/>
    </View>
  </View>
</View>
"""

project = ls.projects.create(
    title='Runway Data Correction',
    label_config=label_config
)

all_tasks = []
fields_to_prefill = [
    "name", "reference_code", "category", "subcategory", 
    "primary_color", "secondary_colors", "pattern", 
    "primary_outer_material", "secondary_outer_materials", "notes"
]

# Only three for testing
for look_id, filenames in list(groups.items())[:3]:
    print(f"Processing Look {look_id}...")
    ai_items = analyze_look(look_id, filenames)
    
    ls_image_data = [f"data:image/jpeg;base64,{encode_image(f)}" for f in filenames]

    for item in ai_items:
        task = {
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
        for i in range(max_images):
            task[f"image_{i}"] = ""

        for i, data_string in enumerate(ls_image_data):
            task[f"image_{i}"] = data_string

        all_tasks.append(task)

ls.projects.import_tasks(
    id=project.id, 
    request=all_tasks,
    preannotated_from_fields=fields_to_prefill
)