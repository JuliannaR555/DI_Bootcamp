# Smart Data Scout - Complete Setup Guide

## Overview

Smart Data Scout is an advanced AI-powered data analysis application that demonstrates the power of MCP (Model Context Protocol) ecosystem integration. It combines multiple MCP servers with LLM orchestration to provide intelligent data analysis capabilities.

## Features

### 🔧 Multi-Server Architecture
- **Filesystem Server**: File operations and data management
- **Analytics Server**: Advanced data analysis and visualization
- **Web Search Server**: Real-time web information retrieval (optional)

### 🤖 AI Orchestration  
- **Groq Integration**: Fast cloud-based LLM processing
- **Ollama Support**: Local LLM execution for privacy
- **Intelligent Planning**: LLM-driven task decomposition and execution
- **Error Handling**: Robust retry mechanisms and adaptive replanning

### 📊 Data Analysis Capabilities
- Statistical analysis and summaries
- Correlation analysis and heatmaps
- Data visualization (charts, plots, graphs)
- Trend analysis and insights generation
- Automated report generation

### 🖥️ User Interface
- **Streamlit Web App**: Interactive chat interface
- **Real-time Logging**: Comprehensive execution monitoring
- **Tool Explorer**: Browse available MCP tools
- **Data Explorer**: Preview and analyze data files

## Quick Start (Recommended)

### Windows Users
1. **Run the quick start script**:
   ```cmd
   quick_start.bat
   ```

2. **Configure your LLM backend**:
   - Edit `.env` file
   - For Groq: Set `GROQ_API_KEY=your_api_key`
   - For Ollama: Set `LLM_BACKEND=ollama`

3. **Launch the application**:
   ```cmd
   streamlit run app.py
   ```

### Manual Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Create sample data** (optional):
   ```bash
   python scripts/setup.py
   ```

4. **Start the application**:
   ```bash
   streamlit run app.py
   ```

## Configuration

### Environment Variables (.env file)

```bash
# LLM Configuration
LLM_BACKEND=groq  # Options: groq, ollama
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-70b-versatile

# Ollama Configuration (if using local LLM)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Application Settings
LOG_LEVEL=INFO
MAX_RETRIES=3
TIMEOUT_SECONDS=30
```

### Groq Setup
1. Visit [Groq Console](https://console.groq.com/)
2. Create an account and get your API key
3. Set `GROQ_API_KEY` in your `.env` file

### Ollama Setup
1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull a model: `ollama pull llama3`
3. Set `LLM_BACKEND=ollama` in your `.env` file

## Usage Examples

### Data Analysis Tasks
- "Analyze the sales data and create a summary report"
- "Show me correlations in the weather data"
- "Create a visualization of revenue trends over time"
- "Generate a comprehensive report for the latest dataset"

### Web Research + Analysis  
- "Search for AI market trends and create a summary"
- "Find renewable energy statistics and save to CSV"
- "Research competitor data and generate insights"

### Complex Workflows
- "Download data about tech stocks, analyze the trends, and create visualizations"
- "Search for climate data, process it, and generate a detailed report"

## Architecture Details

### MCP Server Integration

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

### Key Components

1. **MCP Orchestrator** (`src/orchestrator.py`)
   - Manages connections to multiple MCP servers
   - Coordinates LLM planning and tool execution
   - Handles error recovery and adaptive replanning

2. **LLM Client** (`src/llm/client.py`)
   - Interfaces with Groq or Ollama APIs
   - Generates execution plans from natural language
   - Provides intelligent error handling suggestions

3. **MCP Server Manager** (`src/mcp/server_manager.py`)
   - Starts and manages MCP server processes
   - Monitors server health and status
   - Handles graceful shutdown

4. **Analytics Server** (`servers/analytics_server.py`)
   - Custom MCP server for data analysis
   - Provides statistical analysis tools
   - Generates visualizations and reports

## Available Tools

### Filesystem Tools
- `read_file`: Read file contents
- `write_file`: Write data to files
- `list_directory`: Browse directory contents

### Analytics Tools
- `analyze_data`: Statistical analysis and insights
- `create_visualization`: Charts and graphs
- `generate_report`: Comprehensive data reports

### Web Search Tools (optional)
- `search`: Web search functionality
- Requires additional API keys and setup

## Data Formats

### Supported Input Formats
- **CSV**: Primary format for data analysis
- **JSON**: Configuration and structured data
- **TXT**: Text-based reports and logs

### Generated Outputs
- **PNG**: Visualizations and charts
- **CSV**: Processed data exports
- **TXT**: Analysis reports
- **JSON**: Structured results

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt
   ```

2. **MCP Server Startup Failures**
   - Check if Node.js is installed for npm-based servers
   - Verify Python environment for custom servers
   - Check logs in `logs/app.log`

3. **LLM Connection Issues**
   - Verify API keys in `.env` file
   - Check internet connection for Groq
   - Ensure Ollama is running for local backend

4. **Streamlit Issues**
   ```bash
   # Update Streamlit
   pip install --upgrade streamlit
   
   # Clear cache
   streamlit cache clear
   ```

### Debug Mode

Enable detailed logging:
```bash
# In .env file
LOG_LEVEL=DEBUG
```

View server status:
```bash
python scripts/start_servers.py
```

## Development

### Project Structure
```
smart-data-scout/
├── app.py                  # Main Streamlit application
├── config.yaml            # Application configuration
├── requirements.txt        # Python dependencies
├── quick_start.bat        # Windows quick start script
├── src/                   # Source code
│   ├── config.py          # Configuration management
│   ├── orchestrator.py    # Main orchestration logic
│   ├── llm/              # LLM client code
│   ├── mcp/              # MCP client and server management
│   ├── ui/               # Streamlit UI components
│   └── utils/            # Utility functions
├── servers/              # Custom MCP servers
│   └── analytics_server.py
├── scripts/              # Setup and utility scripts
├── data/                 # Data files
└── logs/                 # Application logs
```

### Adding New MCP Servers

1. **Configure in config.yaml**:
   ```yaml
   mcp_servers:
     your_server:
       name: "your-server"
       command: "python"
       args: ["path/to/your_server.py"]
   ```

2. **Implement server following MCP protocol**
3. **Register tools in orchestrator**

### Custom Tool Development

See `servers/analytics_server.py` for a complete example of implementing custom MCP tools.

## Performance Optimization

### Recommended Settings
- **Groq**: Fast inference, higher costs
- **Ollama**: Slower inference, free local execution
- **Batch Processing**: Process multiple files together
- **Caching**: Enable Streamlit caching for repeated operations

### Scaling Considerations
- MCP servers can be distributed across machines
- LLM calls can be parallelized for independent tasks
- Data processing can be optimized with chunking

## Security Notes

- API keys are stored in `.env` file (not committed to version control)
- Local file access is restricted to configured directories
- MCP servers run in isolated processes
- All external API calls are logged and monitored

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and demonstration purposes. Check individual dependencies for their respective licenses.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs in `logs/app.log`
3. Create an issue with detailed error information

---

**Smart Data Scout** demonstrates the power of MCP ecosystem integration, showing how multiple specialized servers can be orchestrated by AI to solve complex data analysis tasks through natural language interfaces.
