#!/usr/bin/env python3
"""
Setup script for Smart Data Scout
Creates sample data and initializes the environment
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    """Create sample data files for demonstration"""
    
    print("Creating sample data...")
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Sample sales data
    print("- Creating sales_data.csv...")
    np.random.seed(42)  # For reproducible data
    
    sales_data = {
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'product': ['Product A', 'Product B', 'Product C'] * 33 + ['Product A'],
        'sales': np.random.randint(10, 100, 100),
        'revenue': np.random.normal(1000, 200, 100).round(2)
    }
    sales_df = pd.DataFrame(sales_data)
    sales_df.to_csv(data_dir / "sales_data.csv", index=False)
    
    # Sample weather data  
    print("- Creating weather_data.csv...")
    weather_data = {
        'city': ['New York', 'London', 'Tokyo', 'Sydney'] * 25,
        'temperature': np.random.normal(20, 10, 100).round(1),
        'humidity': np.random.normal(60, 15, 100).round(1),
        'date': pd.date_range('2024-01-01', periods=100, freq='D')
    }
    weather_df = pd.DataFrame(weather_data)
    weather_df.to_csv(data_dir / "weather_data.csv", index=False)
    
    # Sample stock data
    print("- Creating stock_data.csv...")
    stock_data = {
        'date': pd.date_range('2024-01-01', periods=50, freq='D'),
        'symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'] * 10,
        'price': np.random.normal(150, 30, 50).round(2),
        'volume': np.random.randint(1000000, 10000000, 50)
    }
    stock_df = pd.DataFrame(stock_data)
    stock_df.to_csv(data_dir / "stock_data.csv", index=False)
    
    print(f"✅ Sample data created successfully in {data_dir}/")
    print(f"   - sales_data.csv ({len(sales_df)} rows)")
    print(f"   - weather_data.csv ({len(weather_df)} rows)")
    print(f"   - stock_data.csv ({len(stock_df)} rows)")

def setup_environment():
    """Setup the environment"""
    
    print("Setting up environment...")
    
    # Create necessary directories
    directories = ['data', 'logs']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")
    
    # Copy .env.example to .env if it doesn't exist
    env_example = Path('.env.example')
    env_file = Path('.env')
    
    if env_example.exists() and not env_file.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from .env.example")
        print("📝 Please edit .env file with your API keys")
    else:
        print("ℹ️  .env file already exists")

if __name__ == "__main__":
    print("🔍 Smart Data Scout - Setup")
    print("=" * 40)
    
    try:
        setup_environment()
        create_sample_data()
        
        print("\n" + "=" * 40)
        print("✅ Setup complete!")
        print("\nNext steps:")
        print("1. Edit .env file with your API keys")
        print("2. Run: streamlit run app.py")
        print("\nFor Groq backend: Your GROQ_API_KEY is already set!")
        print("For Ollama backend: Install Ollama and run 'ollama pull llama3'")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install pandas numpy")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("Please check the error and try again.")
