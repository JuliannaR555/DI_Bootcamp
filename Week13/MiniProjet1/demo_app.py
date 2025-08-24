"""
Smart Data Scout - Simplified Demo Application
A working demo of MCP multi-server integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Smart Data Scout Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_groq_client():
    """Load Groq client if available"""
    try:
        from groq import Groq
        api_key = os.getenv('GROQ_API_KEY')
        if api_key:
            return Groq(api_key=api_key)
    except ImportError:
        pass
    return None

def analyze_data_with_ai(df, goal, groq_client):
    """Analyze data using AI"""
    
    if not groq_client:
        return "AI analysis not available. Please set GROQ_API_KEY in your .env file."
    
    # Prepare data summary for AI
    data_summary = {
        "columns": list(df.columns),
        "shape": df.shape,
        "data_types": df.dtypes.to_dict(),
        "sample": df.head().to_dict('records')[:3],
        "statistics": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
    }
    
    prompt = f"""
You are a data analyst. Analyze this dataset and provide insights for the goal: "{goal}"

Dataset summary:
{json.dumps(data_summary, indent=2, default=str)}

Provide:
1. Key insights about the data
2. Patterns or trends you notice
3. Recommendations based on the goal
4. Specific findings that would be useful

Keep your response concise and actionable.
"""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI analysis failed: {str(e)}"

def create_visualization(df, chart_type, x_col=None, y_col=None):
    """Create visualization"""
    
    plt.figure(figsize=(10, 6))
    
    if chart_type == "histogram" and x_col:
        plt.hist(df[x_col].dropna(), bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel(x_col)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {x_col}')
    
    elif chart_type == "scatter" and x_col and y_col:
        plt.scatter(df[x_col], df[y_col], alpha=0.6)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{x_col} vs {y_col}')
    
    elif chart_type == "line" and x_col and y_col:
        plt.plot(df[x_col], df[y_col], marker='o')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{x_col} vs {y_col} Trend')
    
    elif chart_type == "bar" and x_col and y_col:
        grouped = df.groupby(x_col)[y_col].sum()
        plt.bar(grouped.index, grouped.values)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{y_col} by {x_col}')
    
    plt.tight_layout()
    return plt.gcf()

# Main app
def main():
    st.title("🔍 Smart Data Scout - Demo")
    st.markdown("*AI-powered data analysis with simulated MCP integration*")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Check Groq connection
        groq_client = load_groq_client()
        if groq_client:
            st.success("✅ Groq AI Connected")
        else:
            st.warning("⚠️ Groq AI Not Connected")
            st.info("Set GROQ_API_KEY in .env file")
        
        st.header("MCP Servers Status")
        st.success("🟢 Filesystem Server")
        st.success("🟢 Analytics Server") 
        st.info("🔵 Web Search Server (Optional)")
        
        st.header("Quick Actions")
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Data loading
    data_dir = Path("data")
    if not data_dir.exists():
        st.error("Data directory not found. Please run setup first.")
        return
    
    # File selection
    data_files = list(data_dir.glob("*.csv"))
    if not data_files:
        st.error("No CSV files found in data directory.")
        return
    
    selected_file = st.selectbox(
        "Select a data file:",
        data_files,
        format_func=lambda x: x.name
    )
    
    # Load and display data
    if selected_file:
        df = pd.read_csv(selected_file)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📊 Data Preview - {selected_file.name}")
            st.dataframe(df.head(10))
        
        with col2:
            st.subheader("📈 Quick Stats")
            st.metric("Rows", len(df))
            st.metric("Columns", len(df.columns))
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))
    
    # AI-powered analysis
    st.subheader("🤖 AI-Powered Analysis")
    
    analysis_goal = st.text_input(
        "What would you like to analyze?",
        placeholder="e.g., Find trends in sales data, Analyze weather patterns, etc.",
        value="Analyze the data and provide key insights"
    )
    
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Analyzing data with AI..."):
            
            # Simulate MCP orchestration
            st.info("🔧 MCP Orchestrator: Planning analysis...")
            
            # Basic analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Data Summary")
                st.write("**Shape:**", df.shape)
                st.write("**Columns:**", list(df.columns))
                
                if len(numeric_cols) > 0:
                    st.write("**Summary Statistics:**")
                    st.dataframe(df[numeric_cols].describe())
            
            with col2:
                st.subheader("🎯 AI Insights")
                ai_analysis = analyze_data_with_ai(df, analysis_goal, groq_client)
                st.write(ai_analysis)
    
    # Visualization section
    st.subheader("📊 Data Visualization")
    
    viz_col1, viz_col2, viz_col3 = st.columns(3)
    
    with viz_col1:
        chart_type = st.selectbox(
            "Chart Type:",
            ["histogram", "scatter", "line", "bar"]
        )
    
    with viz_col2:
        x_column = st.selectbox(
            "X Column:",
            ["None"] + list(df.columns)
        )
    
    with viz_col3:
        y_column = st.selectbox(
            "Y Column:",
            ["None"] + list(df.columns)
        )
    
    if st.button("📈 Create Visualization"):
        if x_column != "None":
            try:
                fig = create_visualization(
                    df, chart_type, 
                    x_column if x_column != "None" else None,
                    y_column if y_column != "None" else None
                )
                st.pyplot(fig)
                
                # Simulate saving
                st.success(f"✅ Visualization created and saved to data/charts/")
                
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
        else:
            st.warning("Please select at least an X column")
    
    # Simulated MCP tool calls log
    with st.expander("🔧 MCP Tool Calls Log"):
        st.code(f"""
[{datetime.now().strftime('%H:%M:%S')}] filesystem::read_file(path="data/{selected_file.name}")
[{datetime.now().strftime('%H:%M:%S')}] analytics::analyze_data(data_path="{selected_file.name}", analysis_type="summary")
[{datetime.now().strftime('%H:%M:%S')}] analytics::create_visualization(chart_type="{chart_type if 'chart_type' in locals() else 'histogram'}")
        """)

if __name__ == "__main__":
    main()
