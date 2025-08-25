#!/usr/bin/env python3
"""
Smart Insights MCP Server - Custom Business Intelligence Tool
Provides advanced analytics, predictions, and business insights
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import requests
import warnings
warnings.filterwarnings('ignore')

# Add path for imports
sys.path.append(str(Path(__file__).parent.parent))

def send_response(response):
    """Send JSON-RPC response"""
    print(json.dumps(response))
    sys.stdout.flush()

def handle_tools_list(request):
    """Handle tools/list request"""
    tools = [
        {
            "name": "predict_trends",
            "description": "Predict future trends using machine learning on time series data",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to CSV file with time series data"
                    },
                    "target_column": {
                        "type": "string", 
                        "description": "Column to predict (must be numeric)"
                    },
                    "date_column": {
                        "type": "string",
                        "description": "Date column for time series"
                    },
                    "days_ahead": {
                        "type": "integer",
                        "description": "Number of days to predict into future",
                        "default": 30
                    }
                },
                "required": ["data_path", "target_column"]
            }
        },
        {
            "name": "business_insights",
            "description": "Generate business insights and recommendations from data",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to CSV file with business data"
                    },
                    "business_type": {
                        "type": "string",
                        "description": "Type of business analysis: sales, marketing, financial, operations",
                        "enum": ["sales", "marketing", "financial", "operations"]
                    },
                    "key_metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key metrics to focus on"
                    }
                },
                "required": ["data_path", "business_type"]
            }
        },
        {
            "name": "anomaly_detection",
            "description": "Detect anomalies and outliers in data using statistical methods",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to CSV file"
                    },
                    "target_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to analyze for anomalies"
                    },
                    "sensitivity": {
                        "type": "number",
                        "description": "Sensitivity level (1-10, higher = more sensitive)",
                        "default": 5
                    }
                },
                "required": ["data_path"]
            }
        },
        {
            "name": "competitive_analysis",
            "description": "Perform competitive analysis by comparing metrics across categories",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to CSV file with comparative data"
                    },
                    "category_column": {
                        "type": "string",
                        "description": "Column that defines categories/competitors"
                    },
                    "metric_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metrics to compare"
                    }
                },
                "required": ["data_path", "category_column", "metric_columns"]
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
        if tool_name == 'predict_trends':
            result = predict_trends(arguments)
        elif tool_name == 'business_insights':
            result = business_insights(arguments)
        elif tool_name == 'anomaly_detection':
            result = anomaly_detection(arguments)
        elif tool_name == 'competitive_analysis':
            result = competitive_analysis(arguments)
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
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    return data_dir / relative_path

def predict_trends(arguments):
    """Predict future trends using machine learning"""
    data_path = arguments.get('data_path')
    target_column = arguments.get('target_column')
    date_column = arguments.get('date_column')
    days_ahead = arguments.get('days_ahead', 30)
    
    full_path = get_data_path(data_path)
    
    if not full_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Read data
    df = pd.read_csv(full_path)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    # Prepare data for ML
    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
        # Use date as feature
        df['days_since_start'] = (df[date_column] - df[date_column].min()).dt.days
        X = df[['days_since_start']].values
    else:
        # Use row index as time feature
        X = np.arange(len(df)).reshape(-1, 1)
    
    y = df[target_column].dropna().values
    X = X[:len(y)]
    
    if len(y) < 5:
        raise ValueError("Need at least 5 data points for prediction")
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    last_x = X[-1][0]
    future_x = np.arange(last_x + 1, last_x + days_ahead + 1).reshape(-1, 1)
    predictions = model.predict(future_x)
    
    # Calculate confidence metrics
    train_score = model.score(X, y)
    residuals = y - model.predict(X)
    std_error = np.std(residuals)
    
    # Generate future dates if date column exists
    if date_column and date_column in df.columns:
        last_date = df[date_column].max()
        future_dates = [
            (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
            for i in range(days_ahead)
        ]
    else:
        future_dates = [f"Period {i+len(df)+1}" for i in range(days_ahead)]
    
    # Trend analysis
    trend_direction = "increasing" if model.coef_[0] > 0 else "decreasing"
    trend_strength = abs(model.coef_[0])
    
    return {
        "predictions": [
            {"date": date, "predicted_value": float(pred), "confidence_interval": [float(pred - std_error), float(pred + std_error)]}
            for date, pred in zip(future_dates, predictions)
        ],
        "model_performance": {
            "r_squared": float(train_score),
            "trend_direction": trend_direction,
            "trend_strength": float(trend_strength),
            "standard_error": float(std_error)
        },
        "summary": f"Predicted {days_ahead} days ahead with {train_score:.2%} accuracy. Trend is {trend_direction}.",
        "target_column": target_column,
        "data_points_used": len(y)
    }

def business_insights(arguments):
    """Generate business insights and recommendations"""
    data_path = arguments.get('data_path')
    business_type = arguments.get('business_type')
    key_metrics = arguments.get('key_metrics', [])
    
    full_path = get_data_path(data_path)
    
    if not full_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(full_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    insights = {
        "business_type": business_type,
        "data_overview": {
            "total_records": len(df),
            "time_period": "Unknown",
            "key_metrics_found": [col for col in key_metrics if col in df.columns]
        },
        "insights": [],
        "recommendations": [],
        "kpis": {}
    }
    
    # Detect time period if possible
    date_cols = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                continue
    
    if date_cols:
        date_col = date_cols[0]
        date_range = df[date_col].max() - df[date_col].min()
        insights["data_overview"]["time_period"] = f"{date_range.days} days"
    
    # Business-specific analysis
    if business_type == "sales":
        # Sales analysis
        if 'sales' in df.columns or 'revenue' in df.columns:
            sales_col = 'sales' if 'sales' in df.columns else 'revenue'
            total_sales = df[sales_col].sum()
            avg_sales = df[sales_col].mean()
            growth_rate = (df[sales_col].iloc[-10:].mean() / df[sales_col].iloc[:10].mean() - 1) * 100 if len(df) > 20 else 0
            
            insights["kpis"] = {
                "total_sales": float(total_sales),
                "average_sales": float(avg_sales),
                "growth_rate_percent": float(growth_rate)
            }
            
            insights["insights"].append(f"Total sales: ${total_sales:,.2f}")
            insights["insights"].append(f"Average sales per period: ${avg_sales:,.2f}")
            
            if growth_rate > 5:
                insights["insights"].append(f"Strong growth rate of {growth_rate:.1f}%")
                insights["recommendations"].append("Continue current sales strategy")
            elif growth_rate < -5:
                insights["insights"].append(f"Declining sales trend: {growth_rate:.1f}%")
                insights["recommendations"].append("Review pricing and marketing strategies")
            
            # Best/worst periods
            if date_cols:
                df_with_date = df.copy()
                df_with_date['month'] = df_with_date[date_col].dt.month
                monthly_sales = df_with_date.groupby('month')[sales_col].mean()
                best_month = monthly_sales.idxmax()
                worst_month = monthly_sales.idxmin()
                
                insights["insights"].append(f"Best performing month: {best_month} (${monthly_sales[best_month]:,.2f})")
                insights["insights"].append(f"Lowest performing month: {worst_month} (${monthly_sales[worst_month]:,.2f})")
    
    elif business_type == "marketing":
        # Marketing analysis
        engagement_cols = [col for col in df.columns if any(term in col.lower() for term in ['click', 'view', 'engagement', 'conversion'])]
        
        if engagement_cols:
            for col in engagement_cols:
                avg_val = df[col].mean()
                insights["kpis"][f"avg_{col}"] = float(avg_val)
                insights["insights"].append(f"Average {col}: {avg_val:.2f}")
        
        insights["recommendations"].append("Focus on high-performing channels")
        insights["recommendations"].append("A/B test underperforming campaigns")
    
    elif business_type == "financial":
        # Financial analysis
        profit_cols = [col for col in df.columns if any(term in col.lower() for term in ['profit', 'margin', 'cost', 'expense'])]
        
        if profit_cols:
            for col in profit_cols:
                total_val = df[col].sum()
                insights["kpis"][f"total_{col}"] = float(total_val)
        
        insights["recommendations"].append("Monitor cash flow trends")
        insights["recommendations"].append("Optimize cost structure")
    
    elif business_type == "operations":
        # Operations analysis
        efficiency_cols = [col for col in df.columns if any(term in col.lower() for term in ['efficiency', 'productivity', 'utilization', 'downtime'])]
        
        if numeric_cols:
            # Calculate operational efficiency metrics
            for col in numeric_cols[:3]:  # Top 3 numeric columns
                variance = df[col].var()
                insights["kpis"][f"{col}_variance"] = float(variance)
        
        insights["recommendations"].append("Standardize operational processes")
        insights["recommendations"].append("Implement continuous improvement")
    
    # General insights for all business types
    if len(numeric_cols) > 1:
        # Correlation insights
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []
        
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        "variables": [numeric_cols[i], numeric_cols[j]],
                        "correlation": float(corr_val)
                    })
        
        if high_corr_pairs:
            insights["insights"].append(f"Found {len(high_corr_pairs)} strong correlations")
            insights["correlations"] = high_corr_pairs
    
    # Data quality insights
    missing_data = df.isnull().sum().sum()
    if missing_data > 0:
        missing_pct = (missing_data / (len(df) * len(df.columns))) * 100
        insights["insights"].append(f"Data quality concern: {missing_pct:.1f}% missing values")
        insights["recommendations"].append("Improve data collection processes")
    
    return insights

def anomaly_detection(arguments):
    """Detect anomalies in data using statistical methods"""
    data_path = arguments.get('data_path')
    target_columns = arguments.get('target_columns', [])
    sensitivity = arguments.get('sensitivity', 5)
    
    full_path = get_data_path(data_path)
    
    if not full_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(full_path)
    
    if not target_columns:
        target_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Adjust threshold based on sensitivity (1-10 scale)
    z_threshold = 3.5 - (sensitivity - 5) * 0.3  # Range: 2.0 to 5.0
    
    anomalies = {
        "detection_method": "Z-Score and IQR",
        "sensitivity_level": sensitivity,
        "threshold_used": z_threshold,
        "columns_analyzed": target_columns,
        "anomalies_found": [],
        "summary": {}
    }
    
    total_anomalies = 0
    
    for col in target_columns:
        if col not in df.columns:
            continue
        
        col_data = df[col].dropna()
        if len(col_data) < 10:
            continue
        
        # Z-score method
        z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
        z_anomalies = df[z_scores > z_threshold].index.tolist()
        
        # IQR method
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_anomalies = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        
        # Combine methods
        combined_anomalies = list(set(z_anomalies + iqr_anomalies))
        
        if combined_anomalies:
            col_anomalies = []
            for idx in combined_anomalies:
                col_anomalies.append({
                    "row_index": int(idx),
                    "value": float(df.loc[idx, col]),
                    "z_score": float(z_scores.loc[idx]) if idx in z_scores.index else None,
                    "deviation_from_mean": float(df.loc[idx, col] - col_data.mean()),
                    "percentile": float((col_data <= df.loc[idx, col]).mean() * 100)
                })
            
            anomalies["anomalies_found"].append({
                "column": col,
                "anomaly_count": len(combined_anomalies),
                "anomaly_percentage": (len(combined_anomalies) / len(df)) * 100,
                "anomalies": col_anomalies[:10]  # Limit to top 10
            })
            
            total_anomalies += len(combined_anomalies)
    
    anomalies["summary"] = {
        "total_anomalies": total_anomalies,
        "overall_anomaly_rate": (total_anomalies / len(df)) * 100 if len(df) > 0 else 0,
        "recommendation": "Investigate anomalies for data quality issues or interesting outliers" if total_anomalies > 0 else "No significant anomalies detected"
    }
    
    return anomalies

def competitive_analysis(arguments):
    """Perform competitive analysis"""
    data_path = arguments.get('data_path')
    category_column = arguments.get('category_column')
    metric_columns = arguments.get('metric_columns', [])
    
    full_path = get_data_path(data_path)
    
    if not full_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(full_path)
    
    if category_column not in df.columns:
        raise ValueError(f"Category column '{category_column}' not found")
    
    # Filter metric columns to only numeric ones that exist
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    valid_metrics = [col for col in metric_columns if col in numeric_cols]
    
    if not valid_metrics:
        valid_metrics = numeric_cols[:3]  # Use first 3 numeric columns
    
    analysis = {
        "category_column": category_column,
        "metrics_analyzed": valid_metrics,
        "competitors": [],
        "rankings": {},
        "insights": [],
        "market_summary": {}
    }
    
    # Group by category and calculate aggregated metrics
    grouped = df.groupby(category_column)[valid_metrics].agg(['mean', 'sum', 'count']).round(2)
    
    # Competitor analysis
    for category in grouped.index:
        competitor_data = {
            "name": str(category),
            "metrics": {},
            "rankings": {},
            "market_share": {}
        }
        
        for metric in valid_metrics:
            mean_val = grouped.loc[category, (metric, 'mean')]
            sum_val = grouped.loc[category, (metric, 'sum')]
            count_val = grouped.loc[category, (metric, 'count')]
            
            competitor_data["metrics"][metric] = {
                "average": float(mean_val) if pd.notna(mean_val) else 0,
                "total": float(sum_val) if pd.notna(sum_val) else 0,
                "data_points": int(count_val) if pd.notna(count_val) else 0
            }
        
        analysis["competitors"].append(competitor_data)
    
    # Calculate rankings for each metric
    for metric in valid_metrics:
        metric_values = [(comp["name"], comp["metrics"][metric]["total"]) for comp in analysis["competitors"]]
        metric_values.sort(key=lambda x: x[1], reverse=True)
        
        analysis["rankings"][metric] = [
            {"rank": i+1, "competitor": name, "value": value}
            for i, (name, value) in enumerate(metric_values)
        ]
    
    # Market insights
    total_market = {metric: sum(comp["metrics"][metric]["total"] for comp in analysis["competitors"]) for metric in valid_metrics}
    
    for comp in analysis["competitors"]:
        for metric in valid_metrics:
            if total_market[metric] > 0:
                market_share = (comp["metrics"][metric]["total"] / total_market[metric]) * 100
                comp["market_share"][metric] = round(market_share, 2)
    
    # Generate insights
    for metric in valid_metrics:
        leader = analysis["rankings"][metric][0]
        laggard = analysis["rankings"][metric][-1]
        
        analysis["insights"].append(f"{metric} leader: {leader['competitor']} with {leader['value']:,.2f}")
        
        if len(analysis["rankings"][metric]) > 1:
            gap = leader['value'] - laggard['value']
            analysis["insights"].append(f"{metric} gap between leader and laggard: {gap:,.2f}")
    
    # Market concentration
    if len(analysis["competitors"]) > 1:
        for metric in valid_metrics:
            top_3_share = sum(comp["market_share"].get(metric, 0) for comp in sorted(analysis["competitors"], 
                                                                                   key=lambda x: x["market_share"].get(metric, 0), 
                                                                                   reverse=True)[:3])
            analysis["market_summary"][f"{metric}_top3_concentration"] = round(top_3_share, 2)
    
    return analysis

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
    send_response({"status": "Smart Insights MCP Server Started"})
    
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
            send_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error: Invalid JSON"}
            })
        except Exception as e:
            send_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
            })

if __name__ == "__main__":
    main()
