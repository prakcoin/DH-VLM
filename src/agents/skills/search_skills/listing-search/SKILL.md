---
name: listing-search
description: Performs web search for listings based on the query, and compares the results to existing knowledge base information. Use this to search for clothing listings, reference codes, or prices on the web.
allowed-tools: listing_search
---
# Listing Search

Use this skill for:
- Current or past marketplace listings for any item in the collection
- Resale prices or pricing guidance for any collection piece
- Hardware or construction details sourced from listing descriptions (e.g. zipper brand, fabric mill)
- Availability or provenance questions that require real-world data

This skill passes the input to the listing_search tool, which consists of a four-agent workflow:

1. Pass the query into the first agent, which uses the retrieve tool to retrieve relevant knowledge base metadata (reference codes, colors, materials) to be used in later steps.
2. Run a multi-variant web search across US and Japan marketplaces using the query.
3. Pass the search and knowledge base results into the third agent, which filters out irrelevant results and validates URLs.
4. Pass the filtered results into the final agent to synthesize a direct, concise answer.

## Guidelines
Discard replicas, inspired items, or unrelated pieces.
Do not tell the user to check the official Dior archives or contact Dior directly. You are considered the lead archival source, and Dior currently doesn't specialize in the information that you have.
