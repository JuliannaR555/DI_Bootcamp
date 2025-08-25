"""
Smart Data Scout - Simple Demo without Groq dependency
For testing when Groq module has installation issues
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="Smart Data Scout - Simple",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def simulate_tool_execution(tool_name, server_name, arguments):
    """Simulate MCP tool execution"""
    time.sleep(0.5)
    
    if server_name == "analytics":
        if tool_name == "analyze_data":
            return {
                "file": arguments.get('data_path', 'sales_data.csv'),
                "rows": 100,
                "columns": 4,
                "summary_statistics": {"sales": {"mean": 54.5, "std": 25.3}},
                "insights": ["Average sales: $54.50", "High variability in daily sales"]
            }
    
    elif server_name == "insights":
        if tool_name == "predict_trends":
            return {
                "predictions": [
                    {"date": "2024-09-01", "predicted_value": 67.2, "confidence_interval": [60.1, 74.3]},
                    {"date": "2024-09-02", "predicted_value": 68.1, "confidence_interval": [61.0, 75.2]}
                ],
                "model_performance": {"r_squared": 0.82, "trend_direction": "increasing"},
                "summary": "Predicted 30 days ahead with 82% accuracy. Trend is increasing."
            }
        elif tool_name == "business_insights":
            return {
                "insights": [
                    "Strong growth rate of 12.3%",
                    "Best performing month: March ($892.50)",
                    "Revenue increasing steadily"
                ],
                "recommendations": [
                    "Continue current sales strategy",
                    "Focus marketing on high-performing products"
                ],
                "kpis": {"total_sales": 5450.0, "growth_rate_percent": 12.3}
            }
    
    elif server_name == "enrichment":
        if tool_name == "fetch_market_data":
            return {
                "symbols_fetched": 3,
                "records_created": 10,
                "sample_data": [
                    {"symbol": "AAPL", "price": 175.23, "change_percent": 1.2},
                    {"symbol": "GOOGL", "price": 2745.67, "change_percent": -0.8}
                ]
            }
    
    return {"result": f"Executed {tool_name} on {server_name}", "success": True}

def get_available_tools():
    """Define available MCP tools"""
    return [
        # Analytics Server Tools
        {"name": "analyze_data", "server": "analytics", "description": "Analyze CSV data and generate statistical insights"},
        {"name": "create_visualization", "server": "analytics", "description": "Create charts and graphs from data"},
        
        # Smart Insights Server Tools (Part 2)
        {"name": "predict_trends", "server": "insights", "description": "Predict future trends using machine learning"},
        {"name": "business_insights", "server": "insights", "description": "Generate business insights and recommendations"},
        {"name": "anomaly_detection", "server": "insights", "description": "Detect anomalies and outliers in data"},
        
        # Data Enrichment Server Tools (Part 2)
        {"name": "fetch_market_data", "server": "enrichment", "description": "Fetch real-time market and economic data"},
        {"name": "data_transformation", "server": "enrichment", "description": "Apply advanced data transformations"},
    ]

def main():
    st.title("🔍 Smart Data Scout - Simple Demo")
    st.markdown("*Testing MCP servers functionality without external dependencies*")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        st.success("✅ Local simulation mode")
        
        st.header("🖥️ MCP Servers Status")
        servers = [
            ("🟢 Analytics Server", "Statistical analysis & visualization"),
            ("🟢 Smart Insights Server", "🆕 AI predictions & business intelligence"),
            ("🟢 Data Enrichment Server", "🆕 External data & transformations"),
        ]
        
        for server, description in servers:
            st.write(server)
            st.caption(description)
    
    # Available tools display
    available_tools = get_available_tools()
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🔧 Available Tools")
        
        # Group tools by server
        servers_tools = {}
        for tool in available_tools:
            server = tool['server']
            if server not in servers_tools:
                servers_tools[server] = []
            servers_tools[server].append(tool)
        
        for server_name, tools in servers_tools.items():
            with st.expander(f"📡 {server_name} ({len(tools)} tools)"):
                for tool in tools:
                    if server_name in ['insights', 'enrichment']:
                        st.write(f"🆕 **{tool['name']}**")
                    else:
                        st.write(f"**{tool['name']}**")
                    st.caption(tool['description'])
    
    with col1:
        st.subheader("🤖 Multi-Server Workflow Demo")
        
        # Pre-defined workflow steps
        workflow_steps = [
            {"tool": "analyze_data", "server": "analytics", "description": "Analyze sales data"},
            {"tool": "predict_trends", "server": "insights", "description": "Predict future trends"},
            {"tool": "business_insights", "server": "insights", "description": "Generate business insights"},
            {"tool": "fetch_market_data", "server": "enrichment", "description": "Fetch market data"},
        ]
        
        if st.button("🚀 Run Multi-Server Workflow", type="primary"):
            st.subheader("⚡ Executing Workflow")
            
            progress_bar = st.progress(0)
            execution_results = []
            
            for i, step in enumerate(workflow_steps):
                # Update progress
                progress_bar.progress((i + 1) / len(workflow_steps))
                
                with st.expander(f"Step {i+1}: {step['description']}", expanded=True):
                    col_step1, col_step2 = st.columns([1, 1])
                    
                    with col_step1:
                        st.write(f"**Tool:** {step['tool']}")
                        st.write(f"**Server:** {step['server']}")
                    
                    with col_step2:
                        with st.spinner(f"Running {step['tool']}..."):
                            result = simulate_tool_execution(
                                step['tool'],
                                step['server'],
                                {}
                            )
                        
                        st.success("✅ Completed")
                        st.json(result)
                        
                        execution_results.append({
                            "step": i + 1,
                            "tool": step['tool'],
                            "server": step['server'],
                            "result": result,
                            "success": True
                        })
            
            # Final summary
            st.subheader("📊 Execution Summary")
            
            col_summary1, col_summary2, col_summary3 = st.columns(3)
            
            with col_summary1:
                st.metric("Steps Executed", len(execution_results))
            
            with col_summary2:
                successful_steps = sum(1 for r in execution_results if r.get('success', False))
                st.metric("Success Rate", f"{successful_steps}/{len(execution_results)}")
            
            with col_summary3:
                unique_servers = len(set(r['server'] for r in execution_results))
                st.metric("Servers Used", unique_servers)
            
            st.success("🎉 Multi-server workflow completed successfully!")
    
    # Information about the full version
    st.markdown("---")
    st.info("""
    **📝 Note:** This is a simplified demo version that works without external API dependencies.
    
    **For the full experience with AI orchestration:**
    1. Configure GROQ_API_KEY in .env file
    2. Run: `streamlit run enhanced_demo.py`
    3. Use natural language to describe complex analysis goals
    """)

if __name__ == "__main__":
    main()
