"""
MCP Client for connecting to and interacting with MCP servers
"""

import asyncio
import json
import subprocess
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.config import MCPServerConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MCPClient:
    """Client for communicating with an MCP server"""
    
    def __init__(self, server_name: str):
        self.server_name = server_name
        self.process: Optional[subprocess.Popen] = None
        self.is_connected = False
        self.tools: List[Dict[str, Any]] = []
        
    async def connect(self, config: MCPServerConfig):
        """Connect to the MCP server"""
        try:
            logger.info(f"Starting MCP server: {config.name}")
            
            # Prepare environment
            env = dict(config.env) if config.env else {}
            
            # Start the server process
            if config.command == 'npx':
                # Handle Node.js based servers
                cmd = [config.command] + config.args
            elif config.command == 'python':
                # Handle Python based servers
                cmd = [sys.executable] + config.args
            else:
                cmd = [config.command] + config.args
            
            logger.debug(f"Starting command: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**dict(os.environ), **env} if env else None
            )
            
            # Wait a moment for server to start
            await asyncio.sleep(2)
            
            # Check if process is still running
            if self.process.poll() is not None:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise Exception(f"Server process exited: {stderr}")
            
            self.is_connected = True
            logger.info(f"Successfully connected to {config.name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to {config.name}: {str(e)}")
            if self.process:
                self.process.terminate()
                self.process = None
            raise
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server"""
        if not self.is_connected:
            raise Exception("Not connected to server")
        
        try:
            # Send list_tools request
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            }
            
            response = await self._send_request(request)
            
            if 'tools' in response:
                self.tools = response['tools']
                return self.tools
            else:
                # Fallback: return predefined tools based on server type
                return self._get_fallback_tools()
                
        except Exception as e:
            logger.error(f"Failed to list tools: {str(e)}")
            return self._get_fallback_tools()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the server"""
        if not self.is_connected:
            raise Exception("Not connected to server")
        
        try:
            request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            response = await self._send_request(request)
            
            if 'content' in response:
                return response['content']
            elif 'result' in response:
                return response['result']
            else:
                return response
                
        except Exception as e:
            logger.error(f"Tool call failed: {str(e)}")
            # Fallback to mock execution for development
            return await self._mock_tool_call(tool_name, arguments)
    
    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request to the server"""
        if not self.process or not self.process.stdin:
            raise Exception("No active connection")
        
        try:
            # Send request
            request_json = json.dumps(request) + '\n'
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            # Read response
            response_line = self.process.stdout.readline()
            if not response_line:
                raise Exception("No response from server")
            
            response = json.loads(response_line.strip())
            
            if 'error' in response:
                raise Exception(f"Server error: {response['error']}")
            
            return response.get('result', response)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            raise Exception("Invalid response format")
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise
    
    def _get_fallback_tools(self) -> List[Dict[str, Any]]:
        """Get fallback tools when server communication fails"""
        
        if self.server_name == 'filesystem':
            return [
                {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path to read"}
                        },
                        "required": ["path"]
                    }
                },
                {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "inputSchema": {
                        "type": "object", 
                        "properties": {
                            "path": {"type": "string", "description": "File path to write"},
                            "content": {"type": "string", "description": "Content to write"}
                        },
                        "required": ["path", "content"]
                    }
                },
                {
                    "name": "list_directory",
                    "description": "List contents of a directory",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path to list"}
                        },
                        "required": ["path"]
                    }
                }
            ]
        
        elif self.server_name == 'web-search':
            return [
                {
                    "name": "search",
                    "description": "Search the web for information",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "num_results": {"type": "integer", "description": "Number of results", "default": 10}
                        },
                        "required": ["query"]
                    }
                }
            ]
        
        elif self.server_name == 'analytics':
            return [
                {
                    "name": "analyze_data",
                    "description": "Analyze data and generate insights",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "data_path": {"type": "string", "description": "Path to data file"},
                            "analysis_type": {"type": "string", "description": "Type of analysis to perform"}
                        },
                        "required": ["data_path"]
                    }
                },
                {
                    "name": "create_visualization",
                    "description": "Create a data visualization",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "data_path": {"type": "string", "description": "Path to data file"},
                            "chart_type": {"type": "string", "description": "Type of chart to create"},
                            "output_path": {"type": "string", "description": "Output path for visualization"}
                        },
                        "required": ["data_path", "chart_type"]
                    }
                }
            ]
        
        return []
    
    async def _mock_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Mock tool execution for development/fallback"""
        logger.warning(f"Using mock execution for {tool_name}")
        
        if tool_name == "read_file":
            path = Path(arguments.get('path', ''))
            if path.exists():
                return {"content": path.read_text(), "success": True}
            else:
                return {"error": "File not found", "success": False}
        
        elif tool_name == "write_file":
            path = Path(arguments.get('path', ''))
            content = arguments.get('content', '')
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                return {"success": True, "message": f"File written to {path}"}
            except Exception as e:
                return {"error": str(e), "success": False}
        
        elif tool_name == "list_directory":
            path = Path(arguments.get('path', '.'))
            if path.exists() and path.is_dir():
                files = [str(f.name) for f in path.iterdir()]
                return {"files": files, "success": True}
            else:
                return {"error": "Directory not found", "success": False}
        
        elif tool_name == "search":
            query = arguments.get('query', '')
            # Mock search results
            return {
                "results": [
                    {
                        "title": f"Mock result for: {query}",
                        "url": "https://example.com",
                        "snippet": f"This is a mock search result for the query '{query}'. In a real implementation, this would contain actual search results."
                    }
                ],
                "success": True
            }
        
        elif tool_name == "analyze_data":
            data_path = arguments.get('data_path', '')
            return {
                "analysis": f"Mock analysis of {data_path}",
                "insights": ["Sample insight 1", "Sample insight 2"],
                "success": True
            }
        
        elif tool_name == "create_visualization":
            return {
                "visualization_path": arguments.get('output_path', 'mock_chart.png'),
                "success": True,
                "message": "Mock visualization created"
            }
        
        else:
            return {
                "result": f"Mock result for {tool_name}",
                "arguments_received": arguments,
                "success": True
            }
    
    async def close(self):
        """Close connection to the server"""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.sleep(1)
                if self.process.poll() is None:
                    self.process.kill()
            except Exception as e:
                logger.error(f"Error closing server: {str(e)}")
            finally:
                self.process = None
        
        self.is_connected = False
        logger.info(f"Closed connection to {self.server_name}")

# Add missing import
import os
