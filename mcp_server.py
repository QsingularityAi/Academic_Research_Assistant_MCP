#!/usr/bin/env python3
"""
Academic Research Assistant MCP Server

This script starts a Model Context Protocol (MCP) server that provides
tools for academic research, including paper search, citation analysis,
and research gap identification.
"""
import os
import argparse
import asyncio
import logging
from dotenv import load_dotenv
from mcp import Server, ToolRegistry, StdioConnection, HttpConnection

# Import our MCP tools
from mcp_tools.paper_search_tool import PaperSearchTool
from mcp_tools.citation_analysis_tool import CitationAnalysisTool
from mcp_tools.paper_analysis_tool import PaperAnalysisTool
from mcp_tools.bibliography_tool import BibliographyTool
from mcp_tools.research_gap_tool import ResearchGapTool

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("academic-research-server")

def create_tool_registry():
    """Create and configure the tool registry with all our tools."""
    # Get API keys from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    
    # Log API key status
    logger.info(f"Gemini API Key configured: {bool(gemini_api_key)}")
    logger.info(f"Semantic Scholar API Key configured: {bool(semantic_scholar_api_key)}")
    
    # Initialize tool registry
    registry = ToolRegistry()
    
    # Add all our tools
    registry.register_tool(PaperSearchTool(semantic_scholar_api_key))
    registry.register_tool(CitationAnalysisTool(semantic_scholar_api_key))
    registry.register_tool(PaperAnalysisTool(gemini_api_key, semantic_scholar_api_key))
    registry.register_tool(BibliographyTool(semantic_scholar_api_key))
    registry.register_tool(ResearchGapTool(gemini_api_key, semantic_scholar_api_key))
    
    return registry

async def start_stdio_server():
    """Start the MCP server using stdio communication."""
    registry = create_tool_registry()
    connection = StdioConnection()
    server = Server(registry, connection)
    
    logger.info("Starting Academic Research Assistant MCP Server (stdio mode)")
    await server.serve()

async def start_http_server(host: str, port: int):
    """Start the MCP server using HTTP communication."""
    registry = create_tool_registry()
    connection = HttpConnection(host, port)
    server = Server(registry, connection)
    
    logger.info(f"Starting Academic Research Assistant MCP Server (HTTP mode) on {host}:{port}")
    await server.serve()

def main():
    parser = argparse.ArgumentParser(description="Academic Research Assistant MCP Server")
    parser.add_argument("--connection_type", choices=["stdio", "http"], default="stdio",
                        help="Connection type (stdio or http)")
    parser.add_argument("--host", default="localhost", help="Host for HTTP server")
    parser.add_argument("--port", type=int, default=8080, help="Port for HTTP server")
    
    args = parser.parse_args()
    
    if args.connection_type == "stdio":
        asyncio.run(start_stdio_server())
    else:
        asyncio.run(start_http_server(args.host, args.port))

if __name__ == "__main__":
    main()
