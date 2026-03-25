---
name: image-input
description: Handles image input by analyzing the image and comparing with relevant knowledge base entries. Only use this skill when a query includes an image.
allowed-tools: get_image_input
---
# Image Input

This skill is for handling any queries that include images. This consists of a three agent workflow:

1. Pass the query into the first agent, which retrieves any relevant images related to the image and query from the image knowledge base.
2. Use the get_image_comparison tool to perform a side-by-side evaluation of the query image against each retrieved archive image, and return the full analysis.
3. Finally, synthesize a final answer based on visual and knowledge base information.

## Guidelines
Only use this tool if the user provides an external image file. 
If no text is provided, analyze the image to determine if it is relevant to Dior Homme AW04 and provide a detailed description. 
If the get_image_input tool cannot find an image, stop all processing immediately.
Make sure to pass the full query to the get_image_input tool.