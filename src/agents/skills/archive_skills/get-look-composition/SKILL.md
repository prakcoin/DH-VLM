---
name: get-look-composition
description: Get the full clothing item composition of a look. Use it when the user asks what a certain look consists of.
allowed-tools: get_look_composition
---
# Look Composition

Simply call the get_look_composition tool with a look number to get the full composition of that look.

## Guidelines
The collection consists of exactly 45 looks (1–45). If a requested look number falls outside this range, inform the user that the look does not exist in the collection.
Never ask the user for a look number first. If it isn't provided by the user, assume they do not know it.
Do not tell the user to check the official Dior archives or contact Dior directly. You are considered the lead archival source, and Dior currently doesn't specialize in the information that you have.