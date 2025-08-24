"""
Startup script for Smart Data Scout MCP servers
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import Config
from src.mcp.server_manager import MCPServerManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def main():
    """Start all MCP servers"""
    
    try:
        # Load configuration
        config = Config()
        
        # Validate configuration
        errors = config.validate()
        if errors:
            logger.error("Configuration errors:")
            for error in errors:
                logger.error(f"  - {error}")
            return 1
        
        # Create server manager
        server_manager = MCPServerManager(config)
        
        # Start servers
        logger.info("Starting MCP servers...")
        await server_manager.start_all_servers()
        
        # Show status
        status = server_manager.get_server_status()
        logger.info("Server status:")
        for server_name, server_status in status.items():
            if server_status['running']:
                logger.info(f"  ✅ {server_name} (PID: {server_status['pid']})")
            else:
                logger.error(f"  ❌ {server_name} (failed to start)")
        
        logger.info("All servers started. Press Ctrl+C to stop.")
        
        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down servers...")
            await server_manager.stop_all_servers()
            logger.info("All servers stopped.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to start servers: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
