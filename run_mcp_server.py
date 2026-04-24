
"""
MCP Server Runner
Run this main Flask application
"""
import threading
from app.services.mcp_server import mcp_server

def start_mcp_server():
    """Start MCP server in a separate thread"""
    print("Starting MCP Server...")
    mcp_server.run()

if __name__ == '__main__':
    start_mcp_server()
