"""
Streamlit UI Components for Smart Data Scout
"""

import streamlit as st
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="Smart Data Scout",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_sidebar():
    """Render the sidebar with configuration and status"""
    
    with st.sidebar:
        st.header("🔍 Smart Data Scout")
        st.markdown("*Multi-MCP Server Integration*")
        
        # Configuration section
        st.subheader("⚙️ Configuration")
        
        # LLM Backend selection
        llm_backend = st.selectbox(
            "LLM Backend",
            ["groq", "ollama"],
            help="Choose your LLM backend"
        )
        
        if llm_backend == "groq":
            st.info("💡 Make sure GROQ_API_KEY is set in your environment")
        else:
            st.info("💡 Make sure Ollama is running locally")
        
        # Server status
        st.subheader("🖥️ Server Status")
        
        if 'orchestrator' in st.session_state:
            orchestrator = st.session_state.orchestrator
            
            if hasattr(orchestrator, 'server_manager'):
                server_status = orchestrator.server_manager.get_server_status()
                
                for server_name, status in server_status.items():
                    icon = "🟢" if status['running'] else "🔴"
                    st.write(f"{icon} {server_name}")
                    
                    if status['running']:
                        st.caption(f"PID: {status['pid']}")
                    else:
                        if st.button(f"Start {server_name}", key=f"start_{server_name}"):
                            st.info("Starting server...")
                            # Trigger server start
                            st.rerun()
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        
        if st.button("🔄 Refresh Connections"):
            if 'orchestrator' in st.session_state:
                asyncio.run(st.session_state.orchestrator.connect_all())
                st.success("Connections refreshed!")
                st.rerun()
        
        if st.button("📊 Show Stats"):
            if 'orchestrator' in st.session_state:
                stats = st.session_state.orchestrator.get_execution_stats()
                st.json(stats)
        
        # Example goals
        st.subheader("💡 Example Goals")
        
        example_goals = [
            "Analyze the sales data and create a summary report",
            "Search for renewable energy trends and save to CSV",
            "Create a visualization of the latest data file",
            "Find information about AI trends and generate insights"
        ]
        
        for goal in example_goals:
            if st.button(goal, key=f"example_{hash(goal)}"):
                st.session_state.user_input = goal
                st.rerun()

def render_main_interface(orchestrator):
    """Render the main chat interface"""
    
    # Chat interface
    st.subheader("💬 Chat Interface")
    
    # Display conversation history
    if st.session_state.messages:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show execution details for assistant messages
                if message["role"] == "assistant" and "execution_details" in message:
                    with st.expander("📋 Execution Details"):
                        details = message["execution_details"]
                        
                        # Show plan
                        if "plan" in details:
                            st.subheader("📝 Execution Plan")
                            plan = details["plan"]
                            st.write(f"**Strategy:** {plan.get('strategy', 'N/A')}")
                            
                            if "steps" in plan:
                                for step in plan["steps"]:
                                    with st.container():
                                        col1, col2 = st.columns([3, 1])
                                        with col1:
                                            st.write(f"**Step {step.get('step', '?')}:** {step.get('description', 'N/A')}")
                                            st.caption(f"Tool: {step.get('tool', 'N/A')} (Server: {step.get('server', 'N/A')})")
                                        with col2:
                                            st.success("✅" if step.get('completed', False) else "⏳")
                        
                        # Show results
                        if "results" in details:
                            st.subheader("📊 Results")
                            results = details["results"]
                            
                            if isinstance(results, list):
                                for j, result in enumerate(results):
                                    st.write(f"**Result {j+1}:**")
                                    if isinstance(result, dict):
                                        st.json(result)
                                    else:
                                        st.write(result)
                            else:
                                st.json(results)
                        
                        # Show execution metrics
                        if "execution_time" in details:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Execution Time", f"{details['execution_time']:.2f}s")
                            with col2:
                                st.metric("Tools Used", details.get('tools_used', 0))
                            with col3:
                                success = details.get('success', False)
                                st.metric("Status", "✅ Success" if success else "❌ Failed")
    
    # Input area
    user_input = st.chat_input("What would you like me to help you with?")
    
    # Handle pre-filled input from sidebar
    if 'user_input' in st.session_state:
        user_input = st.session_state.user_input
        del st.session_state.user_input
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Process with orchestrator
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                try:
                    # Execute the goal
                    result = asyncio.run(orchestrator.execute_goal(user_input))
                    
                    if result.get('success', False):
                        # Display success message
                        summary = result.get('summary', 'Task completed successfully!')
                        st.write(summary)
                        
                        # Add to message history with execution details
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": summary,
                            "execution_details": result
                        })
                        
                        # Show any generated files or visualizations
                        if "results" in result:
                            render_results_visualization(result["results"])
                    
                    else:
                        # Display error
                        error_msg = f"Sorry, I encountered an error: {result.get('error', 'Unknown error')}"
                        st.error(error_msg)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
                        
                        # Show partial results if available
                        if result.get('partial_results'):
                            st.subheader("Partial Results")
                            st.json(result['partial_results'])
                
                except Exception as e:
                    error_msg = f"An unexpected error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

def render_results_visualization(results: List[Any]):
    """Render visualizations for results"""
    
    for i, result in enumerate(results):
        if isinstance(result, dict):
            # Check if result contains chart path
            if 'chart_path' in result:
                chart_path = Path(result['chart_path'])
                if chart_path.exists():
                    st.subheader(f"📊 Generated Visualization {i+1}")
                    st.image(str(chart_path), caption=f"Chart: {result.get('chart_type', 'Unknown')}")
            
            # Check if result contains data analysis
            elif 'summary' in result or 'correlation_matrix' in result:
                st.subheader(f"📈 Data Analysis {i+1}")
                
                if 'summary' in result:
                    st.write("**Statistical Summary:**")
                    df_summary = pd.DataFrame(result['summary'])
                    st.dataframe(df_summary)
                
                if 'correlation_matrix' in result:
                    st.write("**Correlation Matrix:**")
                    corr_df = pd.DataFrame(result['correlation_matrix'])
                    fig = px.imshow(corr_df, text_auto=True, aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'distributions' in result:
                    st.write("**Distributions:**")
                    dist_data = result['distributions']
                    
                    # Create distribution plot
                    cols = st.columns(min(len(dist_data), 3))
                    for j, (col_name, stats) in enumerate(dist_data.items()):
                        with cols[j % 3]:
                            st.metric(f"{col_name} Mean", f"{stats['mean']:.2f}")
                            st.metric(f"{col_name} Std", f"{stats['std']:.2f}")
            
            # Check if result contains search results
            elif 'results' in result and isinstance(result['results'], list):
                st.subheader(f"🔍 Search Results {i+1}")
                
                for search_result in result['results'][:5]:  # Show top 5
                    with st.container():
                        st.write(f"**{search_result.get('title', 'No title')}**")
                        st.write(search_result.get('snippet', 'No description'))
                        if 'url' in search_result:
                            st.link_button("Visit", search_result['url'])
                        st.divider()
            
            # Generic result display
            else:
                with st.expander(f"📋 Result {i+1} Details"):
                    st.json(result)

def render_logs_section():
    """Render the logs section"""
    
    if 'logs' in st.session_state and st.session_state.logs:
        # Filter logs
        log_level = st.selectbox(
            "Log Level Filter",
            ["ALL", "INFO", "WARNING", "ERROR"],
            index=0
        )
        
        filtered_logs = st.session_state.logs
        if log_level != "ALL":
            filtered_logs = [log for log in st.session_state.logs if log_level in log]
        
        # Display logs
        if filtered_logs:
            # Show recent logs first
            recent_logs = filtered_logs[-50:]  # Last 50 logs
            
            log_text = "\n".join(recent_logs)
            st.text_area(
                "Recent Logs",
                value=log_text,
                height=200,
                help="Recent application logs"
            )
            
            # Download logs button
            if st.button("📥 Download Full Logs"):
                log_content = "\n".join(st.session_state.logs)
                st.download_button(
                    label="Download logs.txt",
                    data=log_content,
                    file_name=f"smart_data_scout_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        else:
            st.info("No logs match the selected filter.")
    else:
        st.info("No logs available yet.")

def render_tool_explorer():
    """Render tool explorer interface"""
    
    st.subheader("🔧 Available Tools")
    
    if 'orchestrator' in st.session_state:
        orchestrator = st.session_state.orchestrator
        tools = asyncio.run(orchestrator.get_available_tools())
        
        if tools:
            # Group tools by server
            tools_by_server = {}
            for tool in tools:
                server = tool.get('server', 'Unknown')
                if server not in tools_by_server:
                    tools_by_server[server] = []
                tools_by_server[server].append(tool)
            
            # Display tools by server
            for server_name, server_tools in tools_by_server.items():
                with st.expander(f"🖥️ {server_name} ({len(server_tools)} tools)"):
                    
                    for tool in server_tools:
                        st.subheader(f"🔧 {tool['name']}")
                        st.write(tool.get('description', 'No description available'))
                        
                        # Show parameters
                        if 'parameters' in tool and 'properties' in tool['parameters']:
                            st.write("**Parameters:**")
                            params = tool['parameters']['properties']
                            
                            for param_name, param_info in params.items():
                                param_type = param_info.get('type', 'unknown')
                                param_desc = param_info.get('description', 'No description')
                                required = param_name in tool['parameters'].get('required', [])
                                req_text = " (required)" if required else " (optional)"
                                
                                st.write(f"- `{param_name}` ({param_type}){req_text}: {param_desc}")
                        
                        st.divider()
        else:
            st.warning("No tools available. Check server connections.")
    else:
        st.error("Orchestrator not initialized.")

def render_data_explorer():
    """Render data file explorer"""
    
    st.subheader("📁 Data Files")
    
    data_dir = Path("data")
    if data_dir.exists():
        files = list(data_dir.rglob("*"))
        data_files = [f for f in files if f.is_file()]
        
        if data_files:
            # File selection
            selected_file = st.selectbox(
                "Select a file to preview",
                ["None"] + [str(f.relative_to(data_dir)) for f in data_files]
            )
            
            if selected_file != "None":
                file_path = data_dir / selected_file
                
                # File info
                stat = file_path.stat()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("File Size", f"{stat.st_size / 1024:.1f} KB")
                with col2:
                    st.metric("Modified", datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d"))
                with col3:
                    st.metric("Extension", file_path.suffix)
                
                # File preview
                try:
                    if file_path.suffix.lower() == '.csv':
                        df = pd.read_csv(file_path)
                        st.subheader("📊 CSV Preview")
                        st.dataframe(df.head(10))
                        
                        # Basic stats
                        st.subheader("📈 Basic Statistics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Rows", len(df))
                            st.metric("Columns", len(df.columns))
                        with col2:
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            st.metric("Numeric Columns", len(numeric_cols))
                            st.metric("Missing Values", df.isnull().sum().sum())
                    
                    elif file_path.suffix.lower() in ['.txt', '.md', '.json']:
                        content = file_path.read_text()
                        st.subheader("📄 Text Preview")
                        st.text_area("Content", content[:1000] + ("..." if len(content) > 1000 else ""), height=200)
                    
                    else:
                        st.info("Preview not available for this file type.")
                
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        else:
            st.info("No data files found in the data directory.")
            if st.button("Create Sample Data"):
                create_sample_data()
                st.rerun()
    else:
        st.info("Data directory not found.")
        if st.button("Create Data Directory"):
            data_dir.mkdir(exist_ok=True)
            create_sample_data()
            st.rerun()

def create_sample_data():
    """Create sample data files for demonstration"""
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Sample sales data
    sales_data = {
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'product': ['A', 'B', 'C'] * 34 + ['A', 'B'],
        'sales': pd.np.random.randint(10, 100, 100),
        'revenue': pd.np.random.normal(1000, 200, 100)
    }
    sales_df = pd.DataFrame(sales_data)
    sales_df.to_csv(data_dir / "sales_data.csv", index=False)
    
    # Sample weather data
    weather_data = {
        'city': ['New York', 'London', 'Tokyo', 'Sydney'] * 25,
        'temperature': pd.np.random.normal(20, 10, 100),
        'humidity': pd.np.random.normal(60, 15, 100),
        'date': pd.date_range('2024-01-01', periods=100, freq='D')
    }
    weather_df = pd.DataFrame(weather_data)
    weather_df.to_csv(data_dir / "weather_data.csv", index=False)
    
    st.success("Sample data files created!")

# Export all functions for easy import
__all__ = [
    'setup_page_config',
    'render_sidebar', 
    'render_main_interface',
    'render_logs_section',
    'render_tool_explorer',
    'render_data_explorer',
    'render_results_visualization',
    'create_sample_data'
]
