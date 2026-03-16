---
name: get-image-input
description: When provided an image with the query, use this skill to analyze the image and address the user's query.
allowed-tools: image_reader retrieve
---
# Image Input

This skill is for handling any queries that include images. This consists of a three agent workflow:

1. Pass the query into the first agent, which uses the image_reader tool to format the image path from the query to be used in later steps. 
2. Using the newly formatted query, retrieve any relevant images related to the image and query from the image knowledge base.
3. Finally, synthesize a final answer based on visual and knowledge base information.

## Guidelines
Only use this tool if the user provides an external image file. 
If no text is provided, analyze the image to determine if it is relevant to Dior Homme AW04 and provide a detailed description. 
If the get_image_input tool cannot find an image, stop all processing immediately.
Make sure to pass the full query to the get_image_input tool.
