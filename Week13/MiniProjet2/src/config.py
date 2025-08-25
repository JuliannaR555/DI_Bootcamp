"""
Configuration management for Smart Data Scout
Handles environment variables, YAML config, and MCP server setup
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server"""
    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None
    port: Optional[int] = None
    enabled: bool = True

class LLMConfig(BaseModel):
    """LLM configuration"""
    backend: str = Field(default="groq", description="LLM backend to use")
    api_key: Optional[str] = None
    model: str = "llama-3.1-70b-versatile"
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.1

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}"
    file: str = "logs/app.log"
    rotation: str = "10 MB"
    retention: str = "7 days"

class ErrorHandlingConfig(BaseModel):
    """Error handling configuration"""
    max_retries: int = 3
    retry_delay: float = 2.0
    timeout_seconds: int = 30

class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._load_config()
        self._setup_directories()
    
    def _load_config(self):
        """Load configuration from YAML file and environment variables"""
        
        # Load YAML config if exists
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
        else:
            yaml_config = {}
        
        # MCP Servers Configuration
        self.mcp_servers = self._load_mcp_servers(yaml_config.get('mcp_servers', {}))
        
        # LLM Configuration
        llm_backend = os.getenv('LLM_BACKEND', 'groq')
        llm_config = yaml_config.get('llm', {}).get(llm_backend, {})
        
        self.llm = LLMConfig(
            backend=llm_backend,
            api_key=os.getenv('GROQ_API_KEY') if llm_backend == 'groq' else None,
            model=os.getenv(f'{llm_backend.upper()}_MODEL', llm_config.get('model', 'llama-3.1-70b-versatile')),
            base_url=os.getenv(f'{llm_backend.upper()}_BASE_URL', llm_config.get('base_url')),
            max_tokens=int(os.getenv('MAX_TOKENS', llm_config.get('max_tokens', 4096))),
            temperature=float(os.getenv('TEMPERATURE', llm_config.get('temperature', 0.1)))
        )
        
        # Logging Configuration
        logging_config = yaml_config.get('logging', {})
        self.logging = LoggingConfig(
            level=os.getenv('LOG_LEVEL', logging_config.get('level', 'INFO')),
            file=os.getenv('LOG_FILE', logging_config.get('file', 'logs/app.log'))
        )
        
        # Error Handling Configuration
        error_config = yaml_config.get('error_handling', {})
        self.error_handling = ErrorHandlingConfig(
            max_retries=int(os.getenv('MAX_RETRIES', error_config.get('max_retries', 3))),
            retry_delay=float(os.getenv('RETRY_DELAY', error_config.get('retry_delay', 2.0))),
            timeout_seconds=int(os.getenv('TIMEOUT_SECONDS', error_config.get('timeout_seconds', 30)))
        )
        
        # Application settings
        self.data_dir = Path(os.getenv('DATA_DIR', 'data'))
        self.logs_dir = Path(os.getenv('LOGS_DIR', 'logs'))
    
    def _load_mcp_servers(self, servers_config: Dict[str, Any]) -> List[MCPServerConfig]:
        """Load MCP server configurations"""
        servers = []
        
        # Web Search Server (using Brave Search as example)
        if 'web_search' in servers_config:
            config = servers_config['web_search']
            servers.append(MCPServerConfig(
                name=config.get('name', 'web-search'),
                command=config.get('command', 'npx'),
                args=config.get('args', ['@modelcontextprotocol/server-brave-search']),
                env=config.get('env', {'BRAVE_API_KEY': os.getenv('BRAVE_API_KEY', '')})
            ))
        
        # Filesystem Server
        if 'filesystem' in servers_config:
            config = servers_config['filesystem']
            servers.append(MCPServerConfig(
                name=config.get('name', 'filesystem'),
                command=config.get('command', 'npx'),
                args=config.get('args', ['@modelcontextprotocol/server-filesystem', str(self.data_dir)])
            ))
        
        # Custom Analytics Server
        if 'analytics' in servers_config:
            config = servers_config['analytics']
            servers.append(MCPServerConfig(
                name=config.get('name', 'analytics'),
                command=config.get('command', 'python'),
                args=config.get('args', ['servers/analytics_server.py'])
            ))
        
        # Default servers if none configured
        if not servers:
            servers = [
                MCPServerConfig(
                    name='filesystem',
                    command='npx',
                    args=['@modelcontextprotocol/server-filesystem', str(self.data_dir)]
                ),
                MCPServerConfig(
                    name='analytics',
                    command='python',
                    args=['servers/analytics_server.py']
                )
            ]
        
        return servers
    
    def _setup_directories(self):
        """Create necessary directories"""
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration as dictionary"""
        return self.llm.model_dump()
    
    def get_server_by_name(self, name: str) -> Optional[MCPServerConfig]:
        """Get MCP server configuration by name"""
        for server in self.mcp_servers:
            if server.name == name:
                return server
        return None
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check LLM configuration
        if self.llm.backend == 'groq' and not self.llm.api_key:
            errors.append("GROQ_API_KEY is required when using Groq backend")
        
        # Check MCP servers
        if not self.mcp_servers:
            errors.append("At least one MCP server must be configured")
        
        return errors
