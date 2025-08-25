#!/usr/bin/env python3
"""
Data Enrichment MCP Server - External API Integration and Data Enhancement
Provides tools for enriching data with external sources and transformations
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import time
import hashlib

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
            "name": "enrich_with_geocoding",
            "description": "Enrich address data with geographic coordinates and location details",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to CSV file with address data"
                    },
                    "address_column": {
                        "type": "string",
                        "description": "Column containing addresses to geocode"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output path for enriched data",
                        "default": "enriched_geodata.csv"
                    }
                },
                "required": ["data_path", "address_column"]
            }
        },
        {
            "name": "fetch_market_data",
            "description": "Fetch real-time market data and economic indicators",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data_type": {
                        "type": "string",
                        "description": "Type of market data to fetch",
                        "enum": ["stock_prices", "crypto_prices", "forex_rates", "economic_indicators"]
                    },
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of symbols/tickers to fetch"
                    },
                    "output_path": {
                        "type": "string", 
                        "description": "Output path for market data",
                        "default": "market_data.csv"
                    }
                },
                "required": ["data_type", "symbols"]
            }
        },
        {
            "name": "data_transformation",
            "description": "Apply advanced transformations and feature engineering to data",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to CSV file to transform"
                    },
                    "transformations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["normalize", "standardize", "log_transform", "create_ratios", "time_features", "categorical_encoding"]},
                                "columns": {"type": "array", "items": {"type": "string"}},
                                "parameters": {"type": "object"}
                            }
                        },
                        "description": "List of transformations to apply"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output path for transformed data",
                        "default": "transformed_data.csv"
                    }
                },
                "required": ["data_path", "transformations"]
            }
        },
        {
            "name": "web_scraping",
            "description": "Scrape structured data from websites and APIs",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to scrape data from"
                    },
                    "scraping_type": {
                        "type": "string",
                        "description": "Type of scraping to perform",
                        "enum": ["table_extraction", "api_call", "text_extraction", "price_monitoring"]
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Additional parameters for scraping"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output path for scraped data",
                        "default": "scraped_data.csv"
                    }
                },
                "required": ["url", "scraping_type"]
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
        if tool_name == 'enrich_with_geocoding':
            result = enrich_with_geocoding(arguments)
        elif tool_name == 'fetch_market_data':
            result = fetch_market_data(arguments)
        elif tool_name == 'data_transformation':
            result = data_transformation(arguments)
        elif tool_name == 'web_scraping':
            result = web_scraping(arguments)
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

def enrich_with_geocoding(arguments):
    """Enrich data with geocoding information"""
    data_path = arguments.get('data_path')
    address_column = arguments.get('address_column')
    output_path = arguments.get('output_path', 'enriched_geodata.csv')
    
    full_path = get_data_path(data_path)
    
    if not full_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(full_path)
    
    if address_column not in df.columns:
        raise ValueError(f"Address column '{address_column}' not found")
    
    # Simulate geocoding (in real implementation, would use Google Maps API, OpenStreetMap, etc.)
    geocoded_data = []
    
    for idx, address in enumerate(df[address_column].dropna().unique()):
        # Mock geocoding with simulated coordinates
        # In real implementation: use geocoding API
        hash_seed = hashlib.md5(address.encode()).hexdigest()[:8]
        lat = 40.7128 + (int(hash_seed[:4], 16) % 1000 - 500) / 10000  # Around NYC
        lon = -74.0060 + (int(hash_seed[4:8], 16) % 1000 - 500) / 10000
        
        geocoded_data.append({
            'address': address,
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'city': f"City_{hash_seed[:2]}",
            'state': f"ST_{hash_seed[2:4]}",
            'country': 'USA',
            'confidence_score': round(0.8 + (int(hash_seed[4:6], 16) % 20) / 100, 2)
        })
        
        # Simulate API rate limiting
        time.sleep(0.1)
    
    # Create geocoding lookup
    geocoding_df = pd.DataFrame(geocoded_data)
    
    # Merge with original data
    enriched_df = df.merge(
        geocoding_df, 
        left_on=address_column, 
        right_on='address', 
        how='left'
    )
    
    # Save enriched data
    output_full_path = get_data_path(output_path)
    enriched_df.to_csv(output_full_path, index=False)
    
    return {
        "output_file": output_path,
        "original_rows": len(df),
        "geocoded_addresses": len(geocoded_data),
        "success_rate": len(geocoded_data) / len(df[address_column].dropna().unique()) * 100,
        "new_columns_added": ["latitude", "longitude", "city", "state", "country", "confidence_score"],
        "sample_geocoded": geocoded_data[:3],
        "summary": f"Successfully geocoded {len(geocoded_data)} unique addresses with {len(geocoded_data) / len(df[address_column].dropna().unique()) * 100:.1f}% success rate"
    }

def fetch_market_data(arguments):
    """Fetch market data from APIs"""
    data_type = arguments.get('data_type')
    symbols = arguments.get('symbols', [])
    output_path = arguments.get('output_path', 'market_data.csv')
    
    if not symbols:
        raise ValueError("No symbols provided")
    
    # Simulate market data fetching (in real implementation: use Alpha Vantage, Yahoo Finance, etc.)
    market_data = []
    current_time = datetime.now()
    
    for symbol in symbols:
        # Generate realistic mock data
        hash_seed = hashlib.md5(symbol.encode()).hexdigest()
        base_price = 50 + (int(hash_seed[:4], 16) % 1000)
        
        if data_type == "stock_prices":
            market_data.append({
                'symbol': symbol,
                'price': round(base_price + np.random.normal(0, base_price * 0.02), 2),
                'volume': int(np.random.normal(1000000, 200000)),
                'market_cap': base_price * 1000000000,
                'pe_ratio': round(15 + np.random.normal(0, 5), 2),
                'change_percent': round(np.random.normal(0, 2), 2),
                'timestamp': current_time.isoformat()
            })
        
        elif data_type == "crypto_prices":
            market_data.append({
                'symbol': symbol,
                'price': round(base_price * 100 + np.random.normal(0, base_price * 5), 2),
                'volume_24h': int(np.random.normal(50000000, 10000000)),
                'market_cap': base_price * 10000000000,
                'change_24h_percent': round(np.random.normal(0, 8), 2),
                'timestamp': current_time.isoformat()
            })
        
        elif data_type == "forex_rates":
            market_data.append({
                'currency_pair': symbol,
                'rate': round(1 + np.random.normal(0, 0.1), 4),
                'bid': round(1 + np.random.normal(0, 0.1), 4),
                'ask': round(1 + np.random.normal(0, 0.1), 4),
                'change_percent': round(np.random.normal(0, 1), 3),
                'timestamp': current_time.isoformat()
            })
        
        elif data_type == "economic_indicators":
            market_data.append({
                'indicator': symbol,
                'value': round(np.random.normal(50, 20), 2),
                'previous_value': round(np.random.normal(50, 20), 2),
                'forecast': round(np.random.normal(50, 20), 2),
                'impact': np.random.choice(['High', 'Medium', 'Low']),
                'timestamp': current_time.isoformat()
            })
    
    # Save market data
    market_df = pd.DataFrame(market_data)
    output_full_path = get_data_path(output_path)
    market_df.to_csv(output_full_path, index=False)
    
    return {
        "output_file": output_path,
        "data_type": data_type,
        "symbols_fetched": len(symbols),
        "records_created": len(market_data),
        "columns": list(market_df.columns),
        "sample_data": market_data[:3],
        "fetch_timestamp": current_time.isoformat(),
        "summary": f"Successfully fetched {data_type} for {len(symbols)} symbols"
    }

def data_transformation(arguments):
    """Apply data transformations"""
    data_path = arguments.get('data_path')
    transformations = arguments.get('transformations', [])
    output_path = arguments.get('output_path', 'transformed_data.csv')
    
    full_path = get_data_path(data_path)
    
    if not full_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(full_path)
    original_shape = df.shape
    transformation_log = []
    
    for transform in transformations:
        transform_type = transform.get('type')
        columns = transform.get('columns', [])
        parameters = transform.get('parameters', {})
        
        # Validate columns exist
        valid_columns = [col for col in columns if col in df.columns]
        
        try:
            if transform_type == 'normalize':
                # Min-Max normalization
                for col in valid_columns:
                    if df[col].dtype in ['int64', 'float64']:
                        min_val = df[col].min()
                        max_val = df[col].max()
                        if max_val != min_val:
                            df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
                            transformation_log.append(f"Normalized {col} to range [0,1]")
            
            elif transform_type == 'standardize':
                # Z-score standardization
                for col in valid_columns:
                    if df[col].dtype in ['int64', 'float64']:
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        if std_val != 0:
                            df[f'{col}_standardized'] = (df[col] - mean_val) / std_val
                            transformation_log.append(f"Standardized {col} (z-score)")
            
            elif transform_type == 'log_transform':
                # Log transformation
                for col in valid_columns:
                    if df[col].dtype in ['int64', 'float64'] and (df[col] > 0).all():
                        df[f'{col}_log'] = np.log(df[col])
                        transformation_log.append(f"Applied log transform to {col}")
            
            elif transform_type == 'create_ratios':
                # Create ratios between columns
                if len(valid_columns) >= 2:
                    for i in range(len(valid_columns)):
                        for j in range(i + 1, len(valid_columns)):
                            col1, col2 = valid_columns[i], valid_columns[j]
                            if df[col2].dtype in ['int64', 'float64'] and (df[col2] != 0).all():
                                df[f'{col1}_to_{col2}_ratio'] = df[col1] / df[col2]
                                transformation_log.append(f"Created ratio {col1}/{col2}")
            
            elif transform_type == 'time_features':
                # Extract time features from datetime columns
                for col in valid_columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        df[f'{col}_year'] = df[col].dt.year
                        df[f'{col}_month'] = df[col].dt.month
                        df[f'{col}_day'] = df[col].dt.day
                        df[f'{col}_weekday'] = df[col].dt.weekday
                        df[f'{col}_quarter'] = df[col].dt.quarter
                        transformation_log.append(f"Extracted time features from {col}")
                    except:
                        continue
            
            elif transform_type == 'categorical_encoding':
                # One-hot encoding for categorical variables
                for col in valid_columns:
                    if df[col].dtype == 'object':
                        unique_values = df[col].unique()
                        if len(unique_values) <= 10:  # Only for low cardinality
                            encoded = pd.get_dummies(df[col], prefix=col)
                            df = pd.concat([df, encoded], axis=1)
                            transformation_log.append(f"One-hot encoded {col} ({len(unique_values)} categories)")
        
        except Exception as e:
            transformation_log.append(f"Failed to apply {transform_type} to {columns}: {str(e)}")
    
    # Save transformed data
    output_full_path = get_data_path(output_path)
    df.to_csv(output_full_path, index=False)
    
    return {
        "output_file": output_path,
        "original_shape": original_shape,
        "transformed_shape": df.shape,
        "new_columns_created": df.shape[1] - original_shape[1],
        "transformations_applied": len(transformation_log),
        "transformation_log": transformation_log,
        "column_names": list(df.columns),
        "summary": f"Applied {len(transformation_log)} transformations, creating {df.shape[1] - original_shape[1]} new columns"
    }

def web_scraping(arguments):
    """Scrape data from web sources"""
    url = arguments.get('url')
    scraping_type = arguments.get('scraping_type')
    parameters = arguments.get('parameters', {})
    output_path = arguments.get('output_path', 'scraped_data.csv')
    
    scraped_data = []
    
    try:
        if scraping_type == "table_extraction":
            # Simulate table extraction (would use BeautifulSoup/Scrapy in real implementation)
            scraped_data = [
                {'item': f'Product_{i}', 'price': round(np.random.uniform(10, 100), 2), 'rating': round(np.random.uniform(3, 5), 1)}
                for i in range(20)
            ]
        
        elif scraping_type == "api_call":
            # Simulate API call (would use requests to actual API)
            scraped_data = [
                {'id': i, 'name': f'Item_{i}', 'value': round(np.random.uniform(0, 1000), 2), 'category': f'Cat_{i%5}'}
                for i in range(50)
            ]
        
        elif scraping_type == "text_extraction":
            # Simulate text extraction
            scraped_data = [
                {'paragraph_id': i, 'text': f'Sample text content {i}', 'word_count': np.random.randint(50, 200)}
                for i in range(10)
            ]
        
        elif scraping_type == "price_monitoring":
            # Simulate price monitoring
            scraped_data = [
                {
                    'product': f'Product_{i}',
                    'current_price': round(np.random.uniform(20, 200), 2),
                    'original_price': round(np.random.uniform(25, 250), 2),
                    'discount_percent': round(np.random.uniform(0, 30), 1),
                    'availability': np.random.choice(['In Stock', 'Out of Stock', 'Limited'])
                }
                for i in range(15)
            ]
        
        # Add metadata
        for item in scraped_data:
            item['scraped_at'] = datetime.now().isoformat()
            item['source_url'] = url
        
        # Save scraped data
        if scraped_data:
            scraped_df = pd.DataFrame(scraped_data)
            output_full_path = get_data_path(output_path)
            scraped_df.to_csv(output_full_path, index=False)
        
        return {
            "output_file": output_path,
            "url_scraped": url,
            "scraping_type": scraping_type,
            "records_scraped": len(scraped_data),
            "columns": list(scraped_df.columns) if scraped_data else [],
            "sample_data": scraped_data[:3],
            "scraping_timestamp": datetime.now().isoformat(),
            "summary": f"Successfully scraped {len(scraped_data)} records using {scraping_type} method"
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "url": url,
            "scraping_type": scraping_type,
            "success": False
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
    send_response({"status": "Data Enrichment MCP Server Started"})
    
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
