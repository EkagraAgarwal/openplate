"""Official MCP SDK wiring; stdout is reserved for the stdio protocol."""
try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:
    if exc.name != "mcp.server.fastmcp":
        raise
    # The official SDK v2 renamed FastMCP; support the unpinned `mcp` extra.
    from mcp.server.mcpserver import MCPServer as FastMCP

from .service import RunService


def create_server(root) -> FastMCP:
    service = RunService(root)
    server = FastMCP("platesim-3d")
    for name in ("validate_spec", "plan_run", "run_sample", "get_status", "get_manifest",
                 "cross_section", "height_map", "render", "list_runs"):
        server.tool(name=name)(getattr(service, name))
    return server
