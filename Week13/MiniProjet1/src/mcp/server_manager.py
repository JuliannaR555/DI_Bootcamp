"""
MCP Server Manager - Manages lifecycle of multiple MCP servers
"""

import asyncio
import subprocess
import signal
import sys
import os
from typing import Dict, List, Optional
from pathlib import Path

from src.config import Config, MCPServerConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MCPServerManager:
    """Manages multiple MCP server processes"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processes: Dict[str, subprocess.Popen] = {}
        self.server_configs: Dict[str, MCPServerConfig] = {}
        
        # Store server configs by name
        for server_config in config.mcp_servers:
            self.server_configs[server_config.name] = server_config
    
    async def start_all_servers(self):
        """Start all configured MCP servers"""
        logger.info("Starting all MCP servers...")
        
        for server_config in self.config.mcp_servers:
            if server_config.enabled:
                await self.start_server(server_config)
        
        logger.info(f"Started {len(self.processes)} MCP servers")
    
    async def start_server(self, config: MCPServerConfig):
        """Start a single MCP server"""
        logger.info(f"Starting MCP server: {config.name}")
        
        try:
            # Check if server is already running
            if config.name in self.processes:
                logger.warning(f"Server {config.name} is already running")
                return
            
            # Prepare command
            if config.command == 'npx':
                # Check if npx is available
                if not self._check_npx_available():
                    logger.warning("npx not available, using mock server mode")
                    return
                cmd = [config.command] + config.args
            elif config.command == 'python':
                cmd = [sys.executable] + config.args
            else:
                cmd = [config.command] + config.args
            
            # Prepare environment
            env = dict(os.environ)
            if config.env:
                env.update(config.env)
            
            # Create server script if it's our custom analytics server
            if config.name == 'analytics':
                await self._ensure_analytics_server()
            
            logger.debug(f"Starting command: {' '.join(cmd)}")
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=str(Path.cwd())
            )
            
            # Store process
            self.processes[config.name] = process
            
            # Wait a moment and check if process started successfully
            await asyncio.sleep(1)
            
            if process.poll() is not None:
                # Process exited
                stderr = process.stderr.read() if process.stderr else ""
                stdout = process.stdout.read() if process.stdout else ""
                logger.error(f"Server {config.name} failed to start. STDERR: {stderr}, STDOUT: {stdout}")
                del self.processes[config.name]
                
                # Don't raise exception, continue with other servers
                logger.warning(f"Continuing without {config.name} server")
            else:
                logger.info(f"Successfully started {config.name} server (PID: {process.pid})")
                
        except Exception as e:
            logger.error(f"Failed to start server {config.name}: {str(e)}")
            # Don't raise exception, continue with other servers
    
    async def stop_all_servers(self):
        """Stop all running MCP servers"""
        logger.info("Stopping all MCP servers...")
        
        for server_name in list(self.processes.keys()):
            await self.stop_server(server_name)
        
        self.processes.clear()
        logger.info("All MCP servers stopped")
    
    async def stop_server(self, server_name: str):
        """Stop a specific MCP server"""
        if server_name not in self.processes:
            logger.warning(f"Server {server_name} is not running")
            return
        
        process = self.processes[server_name]
        
        try:
            logger.info(f"Stopping server: {server_name}")
            
            # Try graceful shutdown first
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process(process)),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                # Force kill if graceful shutdown failed
                logger.warning(f"Force killing server: {server_name}")
                process.kill()
                await asyncio.create_task(self._wait_for_process(process))
            
            del self.processes[server_name]
            logger.info(f"Successfully stopped {server_name}")
            
        except Exception as e:
            logger.error(f"Error stopping server {server_name}: {str(e)}")
    
    async def _wait_for_process(self, process: subprocess.Popen):
        """Wait for a process to exit"""
        while process.poll() is None:
            await asyncio.sleep(0.1)
    
    def _check_npx_available(self) -> bool:
        """Check if npx is available"""
        try:
            result = subprocess.run(['npx', '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    async def _ensure_analytics_server(self):
        """Ensure the custom analytics server script exists"""
        server_path = Path("servers/analytics_server.py")
        
        if not server_path.exists():
            logger.info("Creating analytics server script...")
            
            server_path.parent.mkdir(exist_ok=True)
            
            # Create a basic analytics server
            server_content = '''#!/usr/bin/env python3
"""
Custom Analytics MCP Server
Provides data analysis and visualization tools
"""

import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def send_response(response):
    """Send JSON-RPC response"""
    print(json.dumps(response))
    sys.stdout.flush()

def handle_request(request):
    """Handle incoming JSON-RPC request"""
    
    if request.get('method') == 'tools/list':
        tools = [
            {
                "name": "analyze_data",
                "description": "Analyze data from CSV file and generate insights",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data_path": {"type": "string", "description": "Path to CSV file"},
                        "analysis_type": {"type": "string", "description": "Type of analysis (summary, correlation, distribution)"}
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
                        "data_path": {"type": "string", "description": "Path to CSV file"},
                        "chart_type": {"type": "string", "description": "Chart type (line, bar, scatter, histogram)"},
                        "x_column": {"type": "string", "description": "X-axis column"},
                        "y_column": {"type": "string", "description": "Y-axis column"},
                        "output_path": {"type": "string", "description": "Output path for chart"}
                    },
                    "required": ["data_path", "chart_type"]
                }
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": request.get('id'),
            "result": {"tools": tools}
        }
    
    elif request.get('method') == 'tools/call':
        params = request.get('params', {})
        tool_name = params.get('name')
        arguments = params.get('arguments', {})
        
        try:
            if tool_name == 'analyze_data':
                result = analyze_data(arguments)
            elif tool_name == 'create_visualization':
                result = create_visualization(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            return {
                "jsonrpc": "2.0",
                "id": request.get('id'),
                "result": {"content": [{"type": "text", "text": json.dumps(result)}]}
            }
            
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request.get('id'),
                "error": {"code": -1, "message": str(e)}
            }
    
    else:
        return {
            "jsonrpc": "2.0",
            "id": request.get('id'),
            "error": {"code": -32601, "message": "Method not found"}
        }

def analyze_data(arguments):
    """Analyze data from CSV file"""
    data_path = arguments.get('data_path')
    analysis_type = arguments.get('analysis_type', 'summary')
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Read data
    df = pd.read_csv(data_path)
    
    result = {
        "file": data_path,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns)
    }
    
    if analysis_type == 'summary':
        result["summary"] = df.describe().to_dict()
        result["data_types"] = df.dtypes.to_dict()
        result["missing_values"] = df.isnull().sum().to_dict()
    
    elif analysis_type == 'correlation':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            result["correlation_matrix"] = df[numeric_cols].corr().to_dict()
    
    elif analysis_type == 'distribution':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        result["distributions"] = {}
        for col in numeric_cols:
            result["distributions"][col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
    
    return result

def create_visualization(arguments):
    """Create data visualization"""
    data_path = arguments.get('data_path')
    chart_type = arguments.get('chart_type', 'bar')
    x_column = arguments.get('x_column')
    y_column = arguments.get('y_column')
    output_path = arguments.get('output_path', 'chart.png')
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Read data
    df = pd.read_csv(data_path)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    if chart_type == 'line' and x_column and y_column:
        plt.plot(df[x_column], df[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
    
    elif chart_type == 'bar' and x_column and y_column:
        plt.bar(df[x_column], df[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
    
    elif chart_type == 'scatter' and x_column and y_column:
        plt.scatter(df[x_column], df[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
    
    elif chart_type == 'histogram' and x_column:
        plt.hist(df[x_column], bins=20)
        plt.xlabel(x_column)
        plt.ylabel('Frequency')
    
    else:
        # Default: show first numeric column distribution
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            plt.hist(df[numeric_cols[0]], bins=20)
            plt.xlabel(numeric_cols[0])
            plt.ylabel('Frequency')
    
    plt.title(f'{chart_type.title()} Chart')
    plt.tight_layout()
    
    # Save chart
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        "chart_path": output_path,
        "chart_type": chart_type,
        "success": True
    }

def main():
    """Main server loop"""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            request = json.loads(line.strip())
            response = handle_request(request)
            send_response(response)
            
        except json.JSONDecodeError:
            continue
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
            }
            send_response(error_response)

if __name__ == "__main__":
    main()
'''
            
            server_path.write_text(server_content)
            logger.info(f"Created analytics server at {server_path}")
    
    def get_server_status(self) -> Dict[str, Dict[str, any]]:
        """Get status of all servers"""
        status = {}
        
        for server_name, process in self.processes.items():
            status[server_name] = {
                'running': process.poll() is None,
                'pid': process.pid,
                'command': self.server_configs[server_name].command,
                'args': self.server_configs[server_name].args
            }
        
        # Add configured but not running servers
        for server_config in self.config.mcp_servers:
            if server_config.name not in status:
                status[server_config.name] = {
                    'running': False,
                    'pid': None,
                    'command': server_config.command,
                    'args': server_config.args,
                    'enabled': server_config.enabled
                }
        
        return status
    
    async def restart_server(self, server_name: str):
        """Restart a specific server"""
        logger.info(f"Restarting server: {server_name}")
        
        # Stop if running
        if server_name in self.processes:
            await self.stop_server(server_name)
        
        # Find server config
        server_config = self.server_configs.get(server_name)
        if server_config:
            await self.start_server(server_config)
        else:
            logger.error(f"No configuration found for server: {server_name}")
    
    def __del__(self):
        """Cleanup on deletion"""
        # Try to stop all processes
        for process in self.processes.values():
            try:
                if process.poll() is None:
                    process.terminate()
            except:
                pass
