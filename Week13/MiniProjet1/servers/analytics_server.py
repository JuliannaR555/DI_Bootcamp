#!/usr/bin/env python3
"""
Custom Analytics MCP Server
Provides data analysis and visualization tools
"""

import json
import sys
import os
from pathlib import Path

# Add path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

def send_response(response):
    """Send JSON-RPC response"""
    print(json.dumps(response))
    sys.stdout.flush()

def send_error(request_id, code, message):
    """Send error response"""
    error_response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message}
    }
    send_response(error_response)

def handle_tools_list(request):
    """Handle tools/list request"""
    tools = [
        {
            "name": "analyze_data",
            "description": "Analyze data from CSV file and generate insights",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string", 
                        "description": "Path to CSV file relative to data directory"
                    },
                    "analysis_type": {
                        "type": "string", 
                        "description": "Type of analysis: summary, correlation, distribution, or trends",
                        "enum": ["summary", "correlation", "distribution", "trends"]
                    }
                },
                "required": ["data_path"]
            }
        },
        {
            "name": "create_visualization",
            "description": "Create data visualization chart",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to CSV file relative to data directory"
                    },
                    "chart_type": {
                        "type": "string",
                        "description": "Chart type: line, bar, scatter, histogram, boxplot, heatmap",
                        "enum": ["line", "bar", "scatter", "histogram", "boxplot", "heatmap"]
                    },
                    "x_column": {
                        "type": "string",
                        "description": "X-axis column name"
                    },
                    "y_column": {
                        "type": "string", 
                        "description": "Y-axis column name"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output path for chart (optional)",
                        "default": "chart.png"
                    }
                },
                "required": ["data_path", "chart_type"]
            }
        },
        {
            "name": "generate_report",
            "description": "Generate a comprehensive data report",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to CSV file relative to data directory"
                    },
                    "report_type": {
                        "type": "string",
                        "description": "Type of report: summary, detailed, or executive",
                        "enum": ["summary", "detailed", "executive"]
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output path for report (optional)",
                        "default": "report.txt"
                    }
                },
                "required": ["data_path"]
            }
        }
    ]
    
    return {
        "jsonrpc": "2.0",
        "id": request.get('id'),
        "result": {"tools": tools}
    }

def handle_tool_call(request):
    """Handle tools/call request"""
    params = request.get('params', {})
    tool_name = params.get('name')
    arguments = params.get('arguments', {})
    
    try:
        if not PANDAS_AVAILABLE:
            raise Exception("pandas, numpy, and matplotlib are required but not installed")
        
        if tool_name == 'analyze_data':
            result = analyze_data(arguments)
        elif tool_name == 'create_visualization':
            result = create_visualization(arguments)
        elif tool_name == 'generate_report':
            result = generate_report(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        return {
            "jsonrpc": "2.0",
            "id": request.get('id'),
            "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
        }
        
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request.get('id'),
            "error": {"code": -1, "message": str(e)}
        }

def get_data_path(relative_path):
    """Get absolute path for data file"""
    # Assume data directory is relative to project root
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    return data_dir / relative_path

def analyze_data(arguments):
    """Analyze data from CSV file"""
    data_path = arguments.get('data_path')
    analysis_type = arguments.get('analysis_type', 'summary')
    
    full_path = get_data_path(data_path)
    
    if not full_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Read data
    try:
        df = pd.read_csv(full_path)
    except Exception as e:
        raise Exception(f"Failed to read CSV file: {str(e)}")
    
    result = {
        "file": data_path,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "analysis_type": analysis_type
    }
    
    if analysis_type == 'summary':
        # Basic statistics
        result["summary_statistics"] = df.describe().to_dict()
        result["missing_values"] = df.isnull().sum().to_dict()
        result["unique_values"] = {col: df[col].nunique() for col in df.columns}
        
        # Sample data
        result["sample_data"] = df.head(5).to_dict('records')
    
    elif analysis_type == 'correlation':
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            result["correlation_matrix"] = corr_matrix.to_dict()
            result["numeric_columns"] = numeric_cols
            
            # Find strong correlations
            strong_corr = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append({
                            "variables": [numeric_cols[i], numeric_cols[j]],
                            "correlation": float(corr_val)
                        })
            result["strong_correlations"] = strong_corr
        else:
            result["message"] = "Insufficient numeric columns for correlation analysis"
    
    elif analysis_type == 'distribution':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        result["distributions"] = {}
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            result["distributions"][col] = {
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "skewness": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis())
            }
    
    elif analysis_type == 'trends':
        # Look for time-based columns
        date_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head())
                    date_cols.append(col)
                except:
                    continue
        
        result["date_columns"] = date_cols
        
        if date_cols:
            # Analyze trends for first date column
            date_col = date_cols[0]
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            df_copy = df_copy.sort_values(date_col)
            
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            trends = {}
            
            for col in numeric_cols:
                # Simple trend analysis
                values = df_copy[col].values
                x = np.arange(len(values))
                correlation = np.corrcoef(x, values)[0, 1] if len(values) > 1 else 0
                
                trends[col] = {
                    "trend_correlation": float(correlation),
                    "trend_direction": "increasing" if correlation > 0.1 else "decreasing" if correlation < -0.1 else "stable",
                    "start_value": float(values[0]) if len(values) > 0 else None,
                    "end_value": float(values[-1]) if len(values) > 0 else None
                }
            
            result["trends"] = trends
        else:
            result["message"] = "No date columns found for trend analysis"
    
    return result

def create_visualization(arguments):
    """Create data visualization"""
    data_path = arguments.get('data_path')
    chart_type = arguments.get('chart_type', 'bar')
    x_column = arguments.get('x_column')
    y_column = arguments.get('y_column')
    output_path = arguments.get('output_path', f'{chart_type}_chart.png')
    
    full_data_path = get_data_path(data_path)
    
    if not full_data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Read data
    try:
        df = pd.read_csv(full_data_path)
    except Exception as e:
        raise Exception(f"Failed to read CSV file: {str(e)}")
    
    # Create output directory
    full_output_path = get_data_path(output_path)
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        if chart_type == 'line':
            if x_column and y_column:
                if x_column in df.columns and y_column in df.columns:
                    ax.plot(df[x_column], df[y_column], marker='o')
                    ax.set_xlabel(x_column)
                    ax.set_ylabel(y_column)
                else:
                    raise ValueError(f"Columns {x_column} or {y_column} not found")
            else:
                # Default: plot first two numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    ax.plot(df[numeric_cols[0]], df[numeric_cols[1]], marker='o')
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel(numeric_cols[1])
                else:
                    raise ValueError("Need at least 2 numeric columns or specify x_column and y_column")
        
        elif chart_type == 'bar':
            if x_column and y_column:
                if x_column in df.columns and y_column in df.columns:
                    # Group by x_column and sum y_column if needed
                    grouped = df.groupby(x_column)[y_column].sum()
                    ax.bar(grouped.index, grouped.values)
                    ax.set_xlabel(x_column)
                    ax.set_ylabel(y_column)
                else:
                    raise ValueError(f"Columns {x_column} or {y_column} not found")
            else:
                # Default: value counts of first categorical column
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    value_counts = df[categorical_cols[0]].value_counts()
                    ax.bar(value_counts.index, value_counts.values)
                    ax.set_xlabel(categorical_cols[0])
                    ax.set_ylabel('Count')
                else:
                    raise ValueError("No categorical columns found or specify x_column and y_column")
        
        elif chart_type == 'scatter':
            if x_column and y_column:
                if x_column in df.columns and y_column in df.columns:
                    ax.scatter(df[x_column], df[y_column], alpha=0.6)
                    ax.set_xlabel(x_column)
                    ax.set_ylabel(y_column)
                else:
                    raise ValueError(f"Columns {x_column} or {y_column} not found")
            else:
                # Default: first two numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel(numeric_cols[1])
                else:
                    raise ValueError("Need at least 2 numeric columns or specify x_column and y_column")
        
        elif chart_type == 'histogram':
            column = x_column or y_column
            if column and column in df.columns:
                ax.hist(df[column].dropna(), bins=20, alpha=0.7, edgecolor='black')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
            else:
                # Default: first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    ax.hist(df[numeric_cols[0]].dropna(), bins=20, alpha=0.7, edgecolor='black')
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel('Frequency')
                else:
                    raise ValueError("No numeric columns found")
        
        elif chart_type == 'boxplot':
            if y_column and y_column in df.columns:
                if x_column and x_column in df.columns:
                    # Grouped boxplot
                    df.boxplot(column=y_column, by=x_column, ax=ax)
                    ax.set_title('')
                    plt.suptitle('')
                else:
                    # Single boxplot
                    ax.boxplot(df[y_column].dropna())
                    ax.set_ylabel(y_column)
                    ax.set_xticklabels([y_column])
            else:
                # Default: boxplot of all numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df[numeric_cols].boxplot(ax=ax)
                else:
                    raise ValueError("No numeric columns found")
        
        elif chart_type == 'heatmap':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlation_matrix = df[numeric_cols].corr()
                im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
                ax.set_xticks(range(len(numeric_cols)))
                ax.set_yticks(range(len(numeric_cols)))
                ax.set_xticklabels(numeric_cols, rotation=45)
                ax.set_yticklabels(numeric_cols)
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
                
                # Add correlation values
                for i in range(len(numeric_cols)):
                    for j in range(len(numeric_cols)):
                        text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black")
            else:
                raise ValueError("Need at least 2 numeric columns for heatmap")
        
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        plt.title(f'{chart_type.title()} Chart - {data_path}')
        plt.tight_layout()
        
        # Save chart
        plt.savefig(full_output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            "chart_path": output_path,
            "chart_type": chart_type,
            "data_file": data_path,
            "success": True,
            "message": f"Created {chart_type} chart at {output_path}"
        }
    
    except Exception as e:
        plt.close()
        raise Exception(f"Failed to create visualization: {str(e)}")

def generate_report(arguments):
    """Generate a comprehensive data report"""
    data_path = arguments.get('data_path')
    report_type = arguments.get('report_type', 'summary')
    output_path = arguments.get('output_path', 'report.txt')
    
    # Get analysis data
    analysis_result = analyze_data({
        'data_path': data_path,
        'analysis_type': 'summary'
    })
    
    # Generate report content
    report_lines = []
    report_lines.append(f"DATA ANALYSIS REPORT")
    report_lines.append(f"=" * 50)
    report_lines.append(f"File: {data_path}")
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Report Type: {report_type}")
    report_lines.append("")
    
    # Basic info
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("-" * 20)
    report_lines.append(f"Rows: {analysis_result['rows']:,}")
    report_lines.append(f"Columns: {analysis_result['columns']}")
    report_lines.append(f"Columns: {', '.join(analysis_result['column_names'])}")
    report_lines.append("")
    
    # Data types
    report_lines.append("DATA TYPES")
    report_lines.append("-" * 20)
    for col, dtype in analysis_result['data_types'].items():
        report_lines.append(f"{col}: {dtype}")
    report_lines.append("")
    
    # Missing values
    if analysis_result['missing_values']:
        missing_any = any(count > 0 for count in analysis_result['missing_values'].values())
        if missing_any:
            report_lines.append("MISSING VALUES")
            report_lines.append("-" * 20)
            for col, count in analysis_result['missing_values'].items():
                if count > 0:
                    percentage = (count / analysis_result['rows']) * 100
                    report_lines.append(f"{col}: {count} ({percentage:.1f}%)")
            report_lines.append("")
    
    if report_type in ['detailed', 'executive']:
        # Summary statistics
        if 'summary_statistics' in analysis_result:
            report_lines.append("SUMMARY STATISTICS")
            report_lines.append("-" * 20)
            
            stats_df = pd.DataFrame(analysis_result['summary_statistics'])
            report_lines.append(stats_df.to_string())
            report_lines.append("")
        
        # Unique values
        report_lines.append("UNIQUE VALUES")
        report_lines.append("-" * 20)
        for col, count in analysis_result['unique_values'].items():
            report_lines.append(f"{col}: {count}")
        report_lines.append("")
    
    if report_type == 'executive':
        # Key insights
        report_lines.append("KEY INSIGHTS")
        report_lines.append("-" * 20)
        
        # High cardinality columns
        high_cardinality = [col for col, count in analysis_result['unique_values'].items() 
                          if count > analysis_result['rows'] * 0.8]
        if high_cardinality:
            report_lines.append(f"High cardinality columns (potential IDs): {', '.join(high_cardinality)}")
        
        # Potential categorical columns
        categorical = [col for col, count in analysis_result['unique_values'].items() 
                      if count < 20 and analysis_result['data_types'][col] == 'object']
        if categorical:
            report_lines.append(f"Categorical columns: {', '.join(categorical)}")
        
        # Data quality score
        missing_ratio = sum(analysis_result['missing_values'].values()) / (analysis_result['rows'] * analysis_result['columns'])
        quality_score = (1 - missing_ratio) * 100
        report_lines.append(f"Data quality score: {quality_score:.1f}%")
        
        report_lines.append("")
    
    # Sample data
    if 'sample_data' in analysis_result:
        report_lines.append("SAMPLE DATA (first 5 rows)")
        report_lines.append("-" * 30)
        sample_df = pd.DataFrame(analysis_result['sample_data'])
        report_lines.append(sample_df.to_string(index=False))
    
    # Write report
    report_content = "\n".join(report_lines)
    
    full_output_path = get_data_path(output_path)
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    full_output_path.write_text(report_content)
    
    return {
        "report_path": output_path,
        "report_type": report_type,
        "data_file": data_path,
        "success": True,
        "lines": len(report_lines),
        "summary": f"Generated {report_type} report with {len(report_lines)} lines"
    }

def handle_request(request):
    """Handle incoming JSON-RPC request"""
    
    try:
        method = request.get('method')
        
        if method == 'tools/list':
            return handle_tools_list(request)
        elif method == 'tools/call':
            return handle_tool_call(request)
        else:
            return {
                "jsonrpc": "2.0",
                "id": request.get('id'),
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }
    
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request.get('id'),
            "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
        }

def main():
    """Main server loop"""
    
    # Send initial message
    send_response({"status": "Analytics MCP Server Started"})
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            request = json.loads(line)
            response = handle_request(request)
            send_response(response)
            
        except json.JSONDecodeError:
            send_error(None, -32700, "Parse error: Invalid JSON")
        except Exception as e:
            send_error(None, -32603, f"Internal error: {str(e)}")

if __name__ == "__main__":
    main()
