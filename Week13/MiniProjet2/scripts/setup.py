#!/usr/bin/env python3
"""
Setup script for Smart Data Scout
Installs dependencies and sets up the environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description="", check=True):
    """Run a command and handle errors"""
    print(f"🔄 {description}")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e.stderr}")
        if check:
            return False
        return True

def main():
    """Main setup function"""
    
    print("🔍 Smart Data Scout - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return 1
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install Python dependencies
    print("\n📦 Installing Python dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("❌ Failed to install Python dependencies")
        return 1
    
    # Check for Node.js (optional for some MCP servers)
    print("\n🟢 Checking Node.js availability...")
    if run_command("node --version", "Checking Node.js", check=False):
        print("✅ Node.js is available")
        
        # Install MCP servers from npm
        print("\n📦 Installing MCP servers...")
        servers_to_install = [
            "@modelcontextprotocol/server-filesystem",
            # Add more servers as needed
        ]
        
        for server in servers_to_install:
            run_command(f"npm install -g {server}", f"Installing {server}", check=False)
    
    else:
        print("⚠️  Node.js not found. Some MCP servers may not be available.")
        print("   Install Node.js from https://nodejs.org/ for full functionality.")
    
    # Create directories
    print("\n📁 Creating directories...")
    directories = ["data", "logs", "servers"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created {directory}/")
    
    # Setup environment file
    print("\n⚙️  Setting up environment...")
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_example.exists() and not env_file.exists():
        env_file.write_text(env_example.read_text())
        print("   ✅ Created .env file from .env.example")
        print("   📝 Please edit .env file with your API keys")
    
    # Test basic functionality
    print("\n🧪 Testing basic functionality...")
    test_command = f"{sys.executable} -c \"from src.config import Config; print('Config loaded successfully')\""
    if run_command(test_command, "Testing configuration loading"):
        print("✅ Basic functionality test passed")
    else:
        print("❌ Basic functionality test failed")
        return 1
    
    # Success message
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python scripts/start_servers.py")
    print("3. Run: streamlit run app.py")
    print("\nFor Groq backend:")
    print("   - Set GROQ_API_KEY in .env file")
    print("\nFor Ollama backend:")
    print("   - Install Ollama: https://ollama.ai/")
    print("   - Run: ollama pull llama3")
    print("   - Set LLM_BACKEND=ollama in .env file")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
