---
name: collection-aggregation
description: Performs map-reduce aggregation across the archive using parallel LLM analysis. Use this for counting items, identifying recurring themes (motifs, colors, silhouettes), or verifying item presence across many looks.
allowed-tools: get_collection_inventory
---
# Collection Analysis

Use this skill to query the archive for specific subsets or a general overview.

## Guidelines
Always pass a natural language `query` describing what you want to know.
Pass `subcategory` and/or `color` if the user specifies an item type or color — this pre-filters the dataset before analysis.
If the user asks for a collection summary or something similar, call `get_collection_inventory` with only the `query` parameter.
If the tool returns no results for specific parameters, retry using synonyms (e.g. 'scarf' instead of 'scarves'). If results are still empty, call the tool with only the `query` parameter to retrieve a full collection summary.
The tool returns a synthesized natural language answer — report it directly without reformatting.