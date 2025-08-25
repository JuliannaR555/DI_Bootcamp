# server.py
from mcp.server.fastmcp import FastMCP

# Create MCP server with name "Demo"
mcp = FastMCP("Demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b

# Register a specific greeting resource
@mcp.resource("greeting://hello")
async def greet_hello() -> str:
    """Return a greeting message for hello."""
    return "Hello, hello!"

@mcp.resource("greeting://{name}")
async def greet(name: str) -> str:
    """Return a greeting message for the given name."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    # Start the server loop
    mcp.run()
