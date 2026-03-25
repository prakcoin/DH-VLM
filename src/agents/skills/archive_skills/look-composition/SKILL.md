---
name: look-composition
description: Retrieves a list of items that make up a look, as well as the look images. Use when a user asks for a look breakdown, or if they ask to see a look.
allowed-tools: get_look_composition
---
# Look Composition

This skill is for obtaining a look composition and images based on a look number. 

1. Pass the look number into the get_look_composition tool to get the full look composition and look images.
2. Return both the look composition and the look images to the user.

## Guidelines
A look number must be provided by the user.
Always provide both the look composition and the images