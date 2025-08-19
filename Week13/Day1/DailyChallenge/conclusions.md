# Daily Challenge 19/08 – My First "Intelligent Bot"

## Overview
This Python program simulates the behavior of an intelligent bot by implementing simplified versions of concepts used in Large Language Models (LLMs) and the Model Context Protocol (MCP). All logic is contained in a single file: `mon_bot_intelligent.py`.

## What the bot does
1. **Handshake (MCP simulation)**  
   The program begins with a "handshake" step to simulate establishing a client-server session.

2. **Tool Discovery (MCP)**  
   It lists the tools available:  
   - `simulated_web_search`  
   - `simulated_llm_summary`  
   - `save_file`

3. **Simulated Web Search**  
   Always returns 3 pre-written articles on everyday health topics.

4. **Simulated LLM Summary**  
   Summarizes the first sentence of each article.  
   During this process, it sends progress notifications (`[NOTIFICATION] ...`) to the user, simulating MCP context behavior.

5. **File Saving**  
   Stores the final briefing in a text file inside the folder `generated_briefings/`, with a filename like:  
   `briefing_everyday_health_2025-08-19.txt`

## Learning Outcomes
Through this project, I understood:
- **Tools as services**: how functions can represent specialized services.  
- **LLMs (simulated)**: a program can generate text from input data.  
- **Context (MCP)**: how real-time notifications help track progress.  
- **Handshake (MCP)**: the concept of an initial connection step.  
- **Tool discovery (MCP)**: how a client can query which tools are available.  
- **Orchestration**: chaining multiple steps to complete a complex workflow.

## Conclusion
This exercise was a practical way to explore advanced AI concepts in a simplified simulation.  
I coded:  
- a client-server style workflow,  
- progress notifications,  
- text generation,  
- file saving.  

Together, these components form the foundation of more complex AI systems built on tools and protocols.
