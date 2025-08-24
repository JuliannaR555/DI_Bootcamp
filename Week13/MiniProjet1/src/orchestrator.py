"""
MCP Orchestrator - Main coordination layer for multiple MCP servers
Handles connection, tool discovery, and LLM-driven orchestration
"""

import asyncio
import json
import subprocess
import signal
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import time

from src.config import Config, MCPServerConfig
from src.utils.logger import get_logger, log_tool_call
from src.llm.client import LLMClient
from src.mcp.client import MCPClient
from src.mcp.server_manager import MCPServerManager

logger = get_logger(__name__)

@dataclass
class ToolCall:
    """Represents a tool call with metadata"""
    tool_name: str
    server_name: str
    arguments: Dict[str, Any]
    timestamp: float
    result: Optional[Any] = None
    error: Optional[str] = None
    duration: Optional[float] = None

class MCPOrchestrator:
    """Main orchestrator for MCP servers and LLM coordination"""
    
    def __init__(self, config: Config):
        self.config = config
        self.server_manager = MCPServerManager(config)
        self.llm_client = LLMClient(config)
        self.mcp_clients: Dict[str, MCPClient] = {}
        self.available_tools: Dict[str, Dict] = {}
        self.execution_history: List[ToolCall] = []
        self.is_running = False
        
        logger.info("MCP Orchestrator initialized")
    
    async def start(self):
        """Start all MCP servers and establish connections"""
        try:
            logger.info("Starting MCP Orchestrator...")
            
            # Start MCP servers
            await self.server_manager.start_all_servers()
            
            # Connect to servers
            await self.connect_all()
            
            # Discover tools
            await self.discover_tools()
            
            self.is_running = True
            logger.info("MCP Orchestrator started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start MCP Orchestrator: {str(e)}")
            raise
    
    async def stop(self):
        """Stop all MCP servers and close connections"""
        try:
            logger.info("Stopping MCP Orchestrator...")
            
            # Close MCP clients
            for client in self.mcp_clients.values():
                await client.close()
            self.mcp_clients.clear()
            
            # Stop servers
            await self.server_manager.stop_all_servers()
            
            self.is_running = False
            logger.info("MCP Orchestrator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping MCP Orchestrator: {str(e)}")
    
    async def connect_all(self):
        """Connect to all configured MCP servers"""
        logger.info("Connecting to MCP servers...")
        
        for server_config in self.config.mcp_servers:
            if not server_config.enabled:
                continue
                
            try:
                client = MCPClient(server_config.name)
                await client.connect(server_config)
                self.mcp_clients[server_config.name] = client
                logger.info(f"Connected to MCP server: {server_config.name}")
                
            except Exception as e:
                logger.error(f"Failed to connect to {server_config.name}: {str(e)}")
    
    async def discover_tools(self):
        """Discover available tools from all connected servers"""
        logger.info("Discovering tools from MCP servers...")
        
        self.available_tools.clear()
        
        for server_name, client in self.mcp_clients.items():
            try:
                tools = await client.list_tools()
                for tool in tools:
                    tool_key = f"{server_name}::{tool['name']}"
                    self.available_tools[tool_key] = {
                        'name': tool['name'],
                        'server': server_name,
                        'description': tool.get('description', ''),
                        'parameters': tool.get('inputSchema', {}),
                        'client': client
                    }
                
                logger.info(f"Discovered {len(tools)} tools from {server_name}")
                
            except Exception as e:
                logger.error(f"Failed to discover tools from {server_name}: {str(e)}")
        
        logger.info(f"Total tools available: {len(self.available_tools)}")
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available tools"""
        return list(self.available_tools.values())
    
    def is_connected(self) -> bool:
        """Check if orchestrator is connected to servers"""
        return self.is_running and len(self.mcp_clients) > 0
    
    async def execute_goal(self, user_goal: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a user goal using LLM orchestration
        
        Args:
            user_goal: Natural language description of what to achieve
            context: Optional context from previous interactions
            
        Returns:
            Dictionary with execution results and metadata
        """
        logger.info(f"Executing goal: {user_goal}")
        
        try:
            # Initialize execution context
            execution_context = {
                'goal': user_goal,
                'available_tools': self.available_tools,
                'history': context.get('history', []) if context else [],
                'data_files': await self._list_data_files(),
                'execution_steps': []
            }
            
            # Plan execution with LLM
            plan = await self.llm_client.plan_execution(execution_context)
            logger.info(f"Generated execution plan with {len(plan.get('steps', []))} steps")
            
            # Execute plan step by step
            results = await self._execute_plan(plan, execution_context)
            
            # Generate final summary
            summary = await self.llm_client.summarize_results(execution_context, results)
            
            return {
                'success': True,
                'goal': user_goal,
                'plan': plan,
                'results': results,
                'summary': summary,
                'execution_time': sum(step.duration or 0 for step in self.execution_history[-len(plan.get('steps', [])):]),
                'tools_used': len(set(step.tool_name for step in self.execution_history[-len(plan.get('steps', [])):]))
            }
            
        except Exception as e:
            logger.error(f"Failed to execute goal: {str(e)}")
            return {
                'success': False,
                'goal': user_goal,
                'error': str(e),
                'partial_results': getattr(self, '_partial_results', None)
            }
    
    async def _execute_plan(self, plan: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a plan step by step"""
        results = []
        
        for i, step in enumerate(plan.get('steps', [])):
            logger.info(f"Executing step {i+1}/{len(plan['steps'])}: {step.get('description', 'Unknown')}")
            
            try:
                # Execute tool call
                tool_result = await self._execute_tool_call(
                    step['tool'],
                    step['arguments'],
                    step.get('server', '')
                )
                
                # Update context with result
                context['execution_steps'].append({
                    'step': i + 1,
                    'tool': step['tool'],
                    'arguments': step['arguments'],
                    'result': tool_result,
                    'success': True
                })
                
                results.append(tool_result)
                
                # Check if we should continue (adaptive planning)
                if await self._should_replan(context, results):
                    logger.info("Replanning based on intermediate results...")
                    new_steps = await self.llm_client.replan_execution(context, results)
                    plan['steps'] = plan['steps'][:i+1] + new_steps
                
            except Exception as e:
                logger.error(f"Step {i+1} failed: {str(e)}")
                
                # Try to recover or continue
                recovery_action = await self._handle_step_failure(step, str(e), context)
                
                context['execution_steps'].append({
                    'step': i + 1,
                    'tool': step['tool'],
                    'arguments': step['arguments'],
                    'error': str(e),
                    'recovery_action': recovery_action,
                    'success': False
                })
                
                if recovery_action.get('continue', False):
                    continue
                else:
                    break
        
        return results
    
    async def _execute_tool_call(self, tool_name: str, arguments: Dict[str, Any], server_name: str = '') -> Any:
        """Execute a single tool call"""
        
        # Find the tool
        tool_key = None
        if server_name:
            tool_key = f"{server_name}::{tool_name}"
        else:
            # Search across all servers
            for key in self.available_tools:
                if key.endswith(f"::{tool_name}"):
                    tool_key = key
                    break
        
        if not tool_key or tool_key not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        tool_info = self.available_tools[tool_key]
        client = tool_info['client']
        
        # Record tool call start
        start_time = time.time()
        tool_call = ToolCall(
            tool_name=tool_name,
            server_name=tool_info['server'],
            arguments=arguments,
            timestamp=start_time
        )
        
        try:
            # Execute tool call
            result = await client.call_tool(tool_name, arguments)
            
            # Record success
            tool_call.result = result
            tool_call.duration = time.time() - start_time
            
            # Log the call
            log_tool_call(tool_name, arguments, result)
            
            self.execution_history.append(tool_call)
            return result
            
        except Exception as e:
            # Record failure
            tool_call.error = str(e)
            tool_call.duration = time.time() - start_time
            
            # Log the error
            log_tool_call(tool_name, arguments, error=str(e))
            
            self.execution_history.append(tool_call)
            raise
    
    async def _should_replan(self, context: Dict[str, Any], results: List[Any]) -> bool:
        """Determine if we should replan based on intermediate results"""
        
        # Simple heuristic: replan if we have errors or unexpected results
        recent_steps = context['execution_steps'][-3:]  # Check last 3 steps
        
        error_count = sum(1 for step in recent_steps if not step['success'])
        if error_count >= 2:
            return True
        
        # Let LLM decide based on results
        should_replan = await self.llm_client.should_replan(context, results)
        return should_replan.get('replan', False)
    
    async def _handle_step_failure(self, step: Dict[str, Any], error: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle step failure with recovery strategies"""
        
        logger.info(f"Handling failure for step: {step['tool']}")
        
        # Retry strategies
        max_retries = self.config.error_handling.max_retries
        retry_delay = self.config.error_handling.retry_delay
        
        # Check if we should retry
        retry_count = step.get('retry_count', 0)
        if retry_count < max_retries:
            logger.info(f"Retrying step (attempt {retry_count + 1}/{max_retries})")
            await asyncio.sleep(retry_delay)
            step['retry_count'] = retry_count + 1
            
            try:
                result = await self._execute_tool_call(
                    step['tool'],
                    step['arguments'],
                    step.get('server', '')
                )
                return {'continue': True, 'retry_success': True, 'result': result}
            except Exception as retry_error:
                logger.error(f"Retry failed: {str(retry_error)}")
        
        # Ask LLM for alternative approach
        alternative = await self.llm_client.suggest_alternative(step, error, context)
        
        return {
            'continue': alternative.get('can_continue', True),
            'alternative_approach': alternative.get('suggestion', ''),
            'modified_arguments': alternative.get('modified_arguments', step['arguments'])
        }
    
    async def _list_data_files(self) -> List[str]:
        """List available data files"""
        try:
            data_dir = Path(self.config.data_dir)
            if data_dir.exists():
                return [str(f.relative_to(data_dir)) for f in data_dir.rglob('*') if f.is_file()]
            return []
        except Exception as e:
            logger.error(f"Failed to list data files: {str(e)}")
            return []
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {'total_calls': 0}
        
        total_calls = len(self.execution_history)
        successful_calls = sum(1 for call in self.execution_history if call.error is None)
        failed_calls = total_calls - successful_calls
        
        avg_duration = sum(call.duration or 0 for call in self.execution_history) / total_calls
        
        tools_used = set(call.tool_name for call in self.execution_history)
        servers_used = set(call.server_name for call in self.execution_history)
        
        return {
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'failed_calls': failed_calls,
            'success_rate': successful_calls / total_calls if total_calls > 0 else 0,
            'average_duration': avg_duration,
            'unique_tools_used': len(tools_used),
            'unique_servers_used': len(servers_used),
            'tools_list': list(tools_used),
            'servers_list': list(servers_used)
        }
