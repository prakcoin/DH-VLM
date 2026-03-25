---
name: knowledge-base-retrieval
description: Retrieves relevant metadata from the knowledge base based on a query. Use this to retrieve relevant item details (reference codes, materials, design features) from a single item.
allowed-tools: retrieve
---
# Knowledge Base Retrieval

This skill is for semantic retrieval based on text queries. Use the retrieve tool and pass only the relevant terms (item name or relevant metadata) rather than the full user query.

## Guidelines
If retrieved results yield a low score, retry with a lower threshold. If results remain below the threshold, return them anyway but state that they are provided with lower confidence.
Do not use this skill for item count queries.
All data within the archival toolset is exclusively from the Dior Homme Autumn/Winter 2004 "Victim of the Crime" collection. There is no data from any other collection.
When declining out-of-scope questions, you must state clearly that your expertise is strictly limited to this collection and offer to assist with any relevant inquiries instead.
If asked to retrieve a look composition, pass the look number given as the query.