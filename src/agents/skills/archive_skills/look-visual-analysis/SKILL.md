---
name: look-visual-analysis
description: Answers queries involving identifying visual characteristics of looks, when the answer isn't present in the knowledge base. Use this skill when knowledge base retrieval is insufficient to answer a specific query.
allowed-tools: get_look_visual_analysis
---
# Visual Confirmation

This skill is for handling any queries that involve visual analysis over details that aren't available in the knowledge base. Pass the query to the get_visual_confirmation tool, which consists of a three agent workflow:

1. Pass the query into the first agent, which uses the retrieve tool to retrieve relevant information to be used in later steps. 
2. Using the look number from the retrieved results, get the look images using the get_look_images tool.
3. Pass the look images into the get_image_details tool to get detailed visual analysis.
4. Finally, synthesize a final answer based on visual and knowledge base information.

## Guidelines
Pass the relevant aspects of the query that need to be analyzed (e.g. an item, look, or feature) to the tool, rather than the full query.
When performing visual analysis, report discrepancies between visual and metadata observations.