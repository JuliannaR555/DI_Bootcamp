"""
Smart Data Scout - Main Streamlit Application
Multi-MCP Server Integration for AI-Powered Data Analysis
"""

import streamlit as st
import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import Config
from src.orchestrator import MCPOrchestrator
from src.ui.components import (
    setup_page_config,
    render_sidebar,
    render_main_interface,
    render_logs_section
)
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def main():
    """Main application entry point"""
    
    # Setup page configuration
    setup_page_config()
    
    # Initialize session state
    if 'orchestrator' not in st.session_state:
        try:
            config = Config()
            st.session_state.orchestrator = MCPOrchestrator(config)
            st.session_state.messages = []
            st.session_state.logs = []
            logger.info("Application initialized successfully")
        except Exception as e:
            st.error(f"Failed to initialize application: {str(e)}")
            logger.error(f"Initialization error: {str(e)}")
            return
    
    # Render sidebar
    render_sidebar()
    
    # Main interface
    st.title("🔍 Smart Data Scout")
    st.markdown("*AI-powered data analysis with multi-MCP server integration*")
    
    # Connection status
    orchestrator = st.session_state.orchestrator
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if orchestrator.is_connected():
                st.success("🟢 MCP Servers Connected")
            else:
                st.error("🔴 MCP Servers Disconnected")
                if st.button("🔄 Reconnect"):
                    with st.spinner("Connecting to MCP servers..."):
                        asyncio.run(orchestrator.connect_all())
                    st.rerun()
        
        with col2:
            available_tools = asyncio.run(orchestrator.get_available_tools())
            st.metric("Available Tools", len(available_tools))
        
        with col3:
            st.metric("Active Sessions", len(st.session_state.messages))
    
    # Main interface
    render_main_interface(orchestrator)
    
    # Logs section (expandable)
    with st.expander("📋 Execution Logs", expanded=False):
        render_logs_section()

async def run_async_main():
    """Async wrapper for main function"""
    main()

if __name__ == "__main__":
    # Run the application
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        st.error(f"Application error: {str(e)}")
