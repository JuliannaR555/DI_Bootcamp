# client.py
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="mcp", args=["run", "server.py"], env=None
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List resources
            print("=== Resources ===")
            resources = await session.list_resources()
            if resources.resources:
                for resource in resources.resources:
                    print(f"Resource: {resource.name} - {resource.uri}")
            else:
                print("No resources found, but server supports resource reading")
            
            # List tools
            print("\n=== Tools ===")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"Tool: {tool.name} - {tool.description}")
            
            # Read greeting://hello
            print("\n=== Reading greeting://hello ===")
            greeting_result = await session.read_resource("greeting://hello")
            for content in greeting_result.contents:
                print(f"Greeting result: {content.text}")
            
            # Read greeting://world (dynamic resource)
            print("\n=== Reading greeting://world ===")
            world_result = await session.read_resource("greeting://world")
            for content in world_result.contents:
                print(f"World greeting result: {content.text}")
            
            # Call add tool with a=1, b=7
            print("\n=== Calling add(1, 7) ===")
            add_result = await session.call_tool("add", {"a": 1, "b": 7})
            for content in add_result.content:
                print(f"Add result: {content.text}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
