"""
LLM Client for orchestrating tool calls and planning
Supports both Groq (cloud) and Ollama (local) backends
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
import httpx

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LLMClient:
    """Client for LLM interactions and orchestration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm_config = config.llm
        self.client = httpx.AsyncClient(timeout=60.0)
        
        # Setup backend-specific configuration
        if self.llm_config.backend == 'groq':
            self.api_key = self.llm_config.api_key
            self.base_url = self.llm_config.base_url or "https://api.groq.com/openai/v1"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        elif self.llm_config.backend == 'ollama':
            self.base_url = self.llm_config.base_url or "http://localhost:11434"
            self.headers = {"Content-Type": "application/json"}
        else:
            raise ValueError(f"Unsupported LLM backend: {self.llm_config.backend}")
    
    async def plan_execution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan execution steps for a given goal"""
        
        system_prompt = """You are an AI assistant that plans and coordinates tool calls to achieve user goals.

You have access to multiple tools across different MCP servers. Your job is to:
1. Analyze the user's goal
2. Break it down into specific, actionable steps
3. Choose the right tools and arguments for each step
4. Create a plan that achieves the goal efficiently

Available tools:
{tools}

Current context:
- Data files available: {data_files}
- Previous conversation: {history}

Return a JSON plan with this structure:
{{
    "goal_analysis": "Analysis of what the user wants to achieve",
    "strategy": "High-level strategy to achieve the goal",
    "steps": [
        {{
            "step": 1,
            "description": "What this step does",
            "tool": "tool_name",
            "server": "server_name", 
            "arguments": {{"arg1": "value1"}},
            "expected_output": "What we expect this step to produce"
        }}
    ],
    "success_criteria": "How to know if we succeeded"
}}

Be specific with tool arguments and ensure each step builds logically on previous steps."""

        user_prompt = f"""
Goal: {context['goal']}

Plan the execution to achieve this goal using the available tools.
Consider the available data files and any previous context.
"""

        tools_description = self._format_tools_for_prompt(context['available_tools'])
        
        formatted_system_prompt = system_prompt.format(
            tools=tools_description,
            data_files=context.get('data_files', []),
            history=context.get('history', [])[-3:]  # Last 3 interactions
        )
        
        try:
            response = await self._chat_completion(formatted_system_prompt, user_prompt)
            plan = self._extract_json_from_response(response)
            
            logger.info(f"Generated execution plan: {plan.get('strategy', 'No strategy')}")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to generate plan: {str(e)}")
            # Return fallback plan
            return self._create_fallback_plan(context)
    
    async def replan_execution(self, context: Dict[str, Any], current_results: List[Any]) -> List[Dict[str, Any]]:
        """Replan execution based on intermediate results"""
        
        system_prompt = """You are replanning execution based on intermediate results.
        
Current progress:
{progress}

Current results:
{results}

Available tools:
{tools}

Generate additional steps to complete the original goal. Return a JSON array of steps:
[
    {{
        "step": N,
        "description": "What this step does",
        "tool": "tool_name", 
        "server": "server_name",
        "arguments": {{"arg1": "value1"}},
        "expected_output": "What we expect"
    }}
]"""

        user_prompt = f"Original goal: {context['goal']}\n\nWhat additional steps are needed?"
        
        tools_description = self._format_tools_for_prompt(context['available_tools'])
        
        formatted_system_prompt = system_prompt.format(
            progress=context.get('execution_steps', []),
            results=current_results,
            tools=tools_description
        )
        
        try:
            response = await self._chat_completion(formatted_system_prompt, user_prompt)
            new_steps = self._extract_json_from_response(response)
            
            if isinstance(new_steps, list):
                return new_steps
            else:
                return new_steps.get('steps', [])
                
        except Exception as e:
            logger.error(f"Failed to replan: {str(e)}")
            return []
    
    async def should_replan(self, context: Dict[str, Any], results: List[Any]) -> Dict[str, Any]:
        """Determine if replanning is needed based on results"""
        
        system_prompt = """Analyze the execution progress and determine if replanning is needed.

Original goal: {goal}
Execution steps so far: {steps}
Current results: {results}

Return JSON:
{{
    "replan": true/false,
    "reason": "why replanning is or isn't needed",
    "confidence": 0.0-1.0
}}"""

        formatted_prompt = system_prompt.format(
            goal=context['goal'],
            steps=context.get('execution_steps', []),
            results=results
        )
        
        try:
            response = await self._chat_completion(formatted_prompt, "Should we replan?")
            decision = self._extract_json_from_response(response)
            return decision
            
        except Exception as e:
            logger.error(f"Failed to determine replanning: {str(e)}")
            return {"replan": False, "reason": "Error in analysis", "confidence": 0.0}
    
    async def suggest_alternative(self, failed_step: Dict[str, Any], error: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest alternative approach for failed step"""
        
        system_prompt = """A tool call failed. Suggest an alternative approach.

Failed step: {step}
Error: {error}
Available tools: {tools}
Context: {context}

Return JSON:
{{
    "can_continue": true/false,
    "suggestion": "alternative approach description",
    "modified_arguments": {{"new_args": "if_needed"}},
    "alternative_tool": "tool_name_if_different"
}}"""

        tools_description = self._format_tools_for_prompt(context['available_tools'])
        
        formatted_prompt = system_prompt.format(
            step=failed_step,
            error=error,
            tools=tools_description,
            context=context['goal']
        )
        
        try:
            response = await self._chat_completion(formatted_prompt, "What should we do instead?")
            suggestion = self._extract_json_from_response(response)
            return suggestion
            
        except Exception as e:
            logger.error(f"Failed to suggest alternative: {str(e)}")
            return {
                "can_continue": True,
                "suggestion": "Retry with same parameters",
                "modified_arguments": failed_step.get('arguments', {})
            }
    
    async def summarize_results(self, context: Dict[str, Any], results: List[Any]) -> str:
        """Generate a summary of execution results"""
        
        system_prompt = """Summarize the execution results for the user.

Original goal: {goal}
Execution steps: {steps}
Results: {results}

Provide a clear, concise summary of:
1. What was accomplished
2. Key findings or outputs
3. Any limitations or issues
4. Recommendations for next steps

Write in a friendly, informative tone."""

        formatted_prompt = system_prompt.format(
            goal=context['goal'],
            steps=context.get('execution_steps', []),
            results=results
        )
        
        try:
            response = await self._chat_completion(formatted_prompt, "Summarize the results.")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to summarize results: {str(e)}")
            return f"Execution completed for goal: {context['goal']}. Check the detailed results above."
    
    async def _chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Send chat completion request to LLM backend"""
        
        if self.llm_config.backend == 'groq':
            return await self._groq_completion(system_prompt, user_prompt)
        elif self.llm_config.backend == 'ollama':
            return await self._ollama_completion(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported backend: {self.llm_config.backend}")
    
    async def _groq_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Send request to Groq API"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": self.llm_config.model,
            "messages": messages,
            "max_tokens": self.llm_config.max_tokens,
            "temperature": self.llm_config.temperature
        }
        
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    async def _ollama_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Send request to Ollama API"""
        
        payload = {
            "model": self.llm_config.model,
            "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
            "stream": False,
            "options": {
                "temperature": self.llm_config.temperature,
                "num_predict": self.llm_config.max_tokens
            }
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result['response']
    
    def _format_tools_for_prompt(self, tools: Dict[str, Any]) -> str:
        """Format tools for inclusion in prompts"""
        
        formatted_tools = []
        for tool_key, tool_info in tools.items():
            tool_desc = f"- {tool_info['name']} ({tool_info['server']}): {tool_info['description']}"
            if tool_info.get('parameters'):
                params = tool_info['parameters'].get('properties', {})
                if params:
                    param_list = ", ".join(f"{k}: {v.get('description', 'No description')}" for k, v in params.items())
                    tool_desc += f"\n  Parameters: {param_list}"
            formatted_tools.append(tool_desc)
        
        return "\n".join(formatted_tools)
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        
        # Try to find JSON in the response
        try:
            # Look for JSON block
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                json_str = response[start:end].strip()
            elif '{' in response and '}' in response:
                # Find the first complete JSON object
                start = response.find('{')
                brace_count = 0
                end = start
                for i, char in enumerate(response[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                json_str = response[start:end]
            else:
                raise ValueError("No JSON found in response")
            
            return json.loads(json_str)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON from response: {str(e)}")
            logger.debug(f"Response was: {response}")
            
            # Return fallback structure
            return {
                "error": "Failed to parse response",
                "raw_response": response,
                "fallback": True
            }
    
    def _create_fallback_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simple fallback plan when LLM planning fails"""
        
        goal = context['goal']
        available_tools = context['available_tools']
        
        # Simple heuristic-based planning
        steps = []
        
        # Check if goal involves data analysis
        if any(word in goal.lower() for word in ['analyze', 'data', 'csv', 'chart', 'visualization']):
            # Add data analysis steps
            if any('filesystem' in tool_key for tool_key in available_tools.keys()):
                steps.append({
                    "step": 1,
                    "description": "List available data files",
                    "tool": "list_directory",
                    "server": "filesystem",
                    "arguments": {"path": "data"},
                    "expected_output": "List of data files"
                })
            
            if any('analytics' in tool_key for tool_key in available_tools.keys()):
                steps.append({
                    "step": 2,
                    "description": "Analyze data file",
                    "tool": "analyze_data",
                    "server": "analytics",
                    "arguments": {"data_path": "data/sample.csv", "analysis_type": "summary"},
                    "expected_output": "Data analysis results"
                })
        
        # Check if goal involves web search
        elif any(word in goal.lower() for word in ['search', 'find', 'web', 'internet']):
            if any('web-search' in tool_key for tool_key in available_tools.keys()):
                steps.append({
                    "step": 1,
                    "description": "Search the web for information",
                    "tool": "search",
                    "server": "web-search",
                    "arguments": {"query": goal, "num_results": 5},
                    "expected_output": "Search results"
                })
        
        # Default: just list available tools
        if not steps:
            steps.append({
                "step": 1,
                "description": "Explore available data",
                "tool": "list_directory",
                "server": "filesystem",
                "arguments": {"path": "."},
                "expected_output": "Available files and directories"
            })
        
        return {
            "goal_analysis": f"Fallback plan for: {goal}",
            "strategy": "Simple heuristic-based approach",
            "steps": steps,
            "success_criteria": "Basic goal completion",
            "fallback": True
        }
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
