# Smart Data Scout - MCP Multi-Server Integration

An intelligent data analysis application that integrates multiple MCP servers to fetch, process, and analyze data with AI orchestration.

## Overview

This application demonstrates the power of MCP ecosystem by integrating:
- **Web Search Server**: For fetching real-time data from the web
- **File System Server**: For reading and writing CSV/data files
- **Custom Analytics Server**: For data processing and insights generation

The LLM orchestrates these tools to achieve complex data analysis goals through natural language instructions.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │  MCP Client      │    │   LLM Backend   │
│   Frontend      │◄──►│  Orchestrator    │◄──►│  (Groq/Ollama)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────┐
        │              MCP Servers                        │
        │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
        │  │    Web      │ │    File     │ │   Custom    ││
        │  │   Search    │ │   System    │ │  Analytics  ││
        │  │   Server    │ │   Server    │ │   Server    ││
        │  └─────────────┘ └─────────────┘ └─────────────┘│
        └─────────────────────────────────────────────────┘
```

## Features

- **Multi-Server Integration**: Seamlessly connects to multiple MCP servers
- **AI Orchestration**: LLM plans and executes tool calls in optimal order
- **Error Handling**: Robust retry mechanisms and fallback strategies
- **Real-time Logging**: Comprehensive observability of all operations
- **Flexible Configuration**: Support for both cloud (Groq) and local (Ollama) LLMs
- **Interactive UI**: Streamlit-based interface for easy interaction

## Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure LLM Backend**:
   ```bash
   # For Groq (recommended)
   set GROQ_API_KEY=your_groq_api_key
   
   # For Ollama (local)
   ollama pull llama3
   ```

3. **Start MCP Servers**:
   ```bash
   python scripts/start_servers.py
   ```

4. **Launch Application**:
   ```bash
   streamlit run app.py
   ```

## Usage Examples

- "Find recent data about renewable energy trends and create a summary report"
- "Search for stock market data, save it to CSV, and generate investment insights"
- "Collect weather data for major cities and create a comparative analysis"

## Configuration

All settings are managed through environment variables and `config.yaml`. See the configuration section for details.
