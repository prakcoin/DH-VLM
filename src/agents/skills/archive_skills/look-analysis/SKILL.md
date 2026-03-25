---
name: look-analysis
description: Answers queries involving individual looks. Use this when asked for a look breakdown, look visual analysis, or general look questions.
allowed-tools: get_look_analysis
---
# Look Analysis

This skill is for handling any queries that involve individual looks. Pass the query to the get_look_analysis tool, which consists of a three agent workflow:

1. Pass the query into the first agent, which uses the retrieve and get_look_composition tools to retrieve relevant information to be used in later steps. 
2. Using the look number from the retrieved results, get the look images using the get_look_images tool.
3. Pass the look images into the get_image_details tool to get detailed visual analysis.
4. Finally, synthesize a final answer based on visual and knowledge base information.

## Guidelines
Pass the relevant aspects of the query that need to be analyzed (e.g. an item, look, or feature) to the tool, rather than the full query.
When performing visual analysis, report discrepancies between visual and metadata observations.