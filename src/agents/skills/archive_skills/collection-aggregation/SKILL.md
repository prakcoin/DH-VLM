---
name: collection-aggregation
description: Performs deterministic aggregation across the archive. Use this for counting items, identifying recurring themes (motifs, colors, silhouettes), or verifying item presence across many looks.
allowed-tools: get_collection_inventory
---
# Collection Analysis

Use this skill to query the archive for specific subsets or a general overview. 

## Guidelines 
You must pass parameters if the user specifies an item type or color.
If the user asks for a collection summary or something similar, call get_collection_inventory with no parameters.
If the tool returns no results for specific parameters, retry using synonyms (e.g. 'scarf' instead of 'scarves'). If results are still empty, call the tool with no parameters to retrieve a full collection summary.
If the tool returns a 'Qty(UniqueLooks)' like '12(10L)', report it as '12 items across 10 distinct looks.'