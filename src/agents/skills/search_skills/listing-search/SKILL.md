---
name: listing-search
description: Performs web search for listings based on the query, and compares the results to existing knowledge base information. Use this to search for clothing listings, reference codes, or prices on the web.
allowed-tools: listing_search
---
# Listing Search

This skill is for searching for current and past listings. Pass the input to the listing_search tool, which consists of a three agent workflow: 

1. Pass the query into the first agent, which uses the retrieve tool to retrieve relevant information to be used in later steps. 
2. Pass the knowledge base results to the second agent, in order to search for current and past listings using the query and knowledge base results.
3. Finally, pass both the search and knowledge base results into the third agent, then filter the search results based on the knowledge base results.

## Guidelines
Discard replicas, inspired items, or unrelated pieces.
Do not tell the user to check the official Dior archives or contact Dior directly. You are considered the lead archival source, and Dior currently doesn't specialize in the information that you have.
