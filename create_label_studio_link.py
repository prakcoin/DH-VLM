import os
import json
import csv

IMAGE_REL_PATH = "data/images"
labels_dir = "data/labels"
images_dir = "data/images"

tasks = []

for csv_file in os.listdir(labels_dir):
    if not csv_file.endswith(".csv"):
        continue

    look_id = csv_file.replace("look", "").replace(".csv", "")

    img_files = [
        f for f in os.listdir(images_dir)
        if f.startswith(f"look{look_id}_")
    ]
    img_files.sort()

    image_paths = [
        f"/data/local-files/?d={IMAGE_REL_PATH}/{img}"
        for img in img_files
    ]

    with open(os.path.join(labels_dir, csv_file), newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            data_dict = {
                "look_number": look_id,

                "name": row.get("Name", ""),
                "reference_code": row.get("Reference Code", ""),
                "category": row.get("Category", ""),
                "subcategory": row.get("Subcategory", ""),
                "primary_color": row.get("Primary Color", ""),
                "secondary_colors": row.get("Secondary Color(s)", ""),
                "pattern": row.get("Pattern", ""),
                "primary_outer_material": row.get("Primary Outer Material", ""),
                "secondary_outer_materials": row.get("Secondary Outer Material(s)", ""),
                "notes": row.get("Additional Notes", "")
            }

            for i in range(5):
                if i < len(image_paths):
                    data_dict[f"image_{i}"] = image_paths[i]
                else:
                    data_dict[f"image_{i}"] = ""

            tasks.append({"data": data_dict})

with open("import_to_ls.json", "w") as f:
    json.dump(tasks, f, indent=2)

print(f"Created {len(tasks)} garment-level tasks.")