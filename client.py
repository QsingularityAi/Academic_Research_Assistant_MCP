#!/usr/bin/env python3
"""
Academic Research Assistant Client

This client interacts with the Academic Research Assistant MCP server
and Gemini to provide a natural language interface for academic research tasks.
"""
import os
import json
import asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
load_dotenv()

# Configure Google Generative AI client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Configure MCP server parameters
server_params = StdioServerParameters(
    command="python mcp_server.py",
    args=["--connection_type", "stdio"],
    env={
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "SEMANTIC_SCHOLAR_API_KEY": os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    },
)

# Helper functions for displaying results
def print_papers(papers):
    """Print paper search results in a formatted way."""
    print("\n📚 PAPERS FOUND 📚")
    print("=" * 80)
    for i, paper in enumerate(papers):
        print(f"{i+1}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   Year: {paper['year']} | Citations: {paper.get('citation_count', 'N/A')}")
        print(f"   Source: {paper['source']} | Venue: {paper.get('venue', 'N/A')}")
        if paper.get('doi'):
            print(f"   DOI: {paper['doi']}")
        if paper.get('url'):
            print(f"   URL: {paper['url']}")
        if i < len(papers) - 1:
            print("-" * 80)

def print_citation_analysis(analysis):
    """Print citation analysis results in a formatted way."""
    print("\n🔍 CITATION NETWORK ANALYSIS 🔍")
    print("=" * 80)
    
    # Print network stats
    stats = analysis.get("network_stats", {})
    print("Network Statistics:")
    print(f"- Nodes: {stats.get('nodes', 0)}")
    print(f"- Edges: {stats.get('edges', 0)}")
    print(f"- Average Citations: {stats.get('average_citations', 0)}")
    
    # Print most cited papers
    print("\nMost Cited Papers:")
    for i, paper in enumerate(stats.get("most_cited_papers", [])):
        print(f"{i+1}. {paper.get('title', 'Unknown')}")
        print(f"   Authors: {paper.get('authors', 'Unknown')}")
        print(f"   Year: {paper.get('year', 'N/A')} | Citations: {paper.get('citation_count', 0)}")
    
    # Print clusters
    print("\nResearch Clusters:")
    for i, cluster in enumerate(analysis.get("clusters", [])):
        print(f"Cluster {i+1}: {cluster.get('name', 'Unknown')}")
        print(f"Size: {cluster.get('size', 0)} papers")
        print("Key Papers:")
        for j, paper in enumerate(cluster.get("key_papers", [])):
            print(f"  - {paper}")
    
    # Save visualization if available
    if "visualization" in analysis:
        with open("citation_network.html", "w") as f:
            f.write(analysis["visualization"])
        print("\nVisualization saved to citation_network.html")

def print_paper_analysis(analysis):
    """Print paper analysis results in a formatted way."""
    print("\n📝 PAPER ANALYSIS 📝")
    print("=" * 80)
    
    # Print paper info
    paper_info = analysis.get("paper_info", {})
    print(f"Title: {paper_info.get('title', 'Unknown')}")
    print(f"Authors: {', '.join(paper_info.get('authors', []))}")
    print(f"Year: {paper_info.get('year', 'N/A')} | Venue: {paper_info.get('venue', 'N/A')}")
    
    # Print summary
    if "summary" in analysis:
        print("\nSUMMARY:")
        print(analysis["summary"])
    
    # Print key findings
    if "key_findings" in analysis and analysis["key_findings"]:
        print("\nKEY FINDINGS:")
        for i, finding in enumerate(analysis["key_findings"]):
            print(f"{i+1}. {finding}")
    
    # Print methodology
    if "methodology" in analysis and analysis["methodology"]:
        print("\nMETHODOLOGY:")
        print(analysis["methodology"])
    
    # Print research gaps
    if "research_gaps" in analysis and analysis["research_gaps"]:
        print("\nRESEARCH GAPS:")
        for i, gap in enumerate(analysis["research_gaps"]):
            print(f"{i+1}. {gap}")
    
    # Print future directions
    if "future_directions" in analysis and analysis["future_directions"]:
        print("\nFUTURE DIRECTIONS:")
        for i, direction in enumerate(analysis["future_directions"]):
            print(f"{i+1}. {direction}")

def print_bibliography(bib_result):
    """Print bibliography results in a formatted way."""
    print("\n📖 BIBLIOGRAPHY 📖")
    print("=" * 80)
    
    if "formatted_references" in bib_result:
        print("Formatted References:")
        for i, ref in enumerate(bib_result["formatted_references"]):
            print(f"{i+1}. {ref}")
            if i < len(bib_result["formatted_references"]) - 1:
                print("-" * 40)
    
    if "bibtex" in bib_result:
        print("\nBibTeX:")
        print(bib_result["bibtex"])
    
    if "validation_issues" in bib_result:
        print("\nValidation Issues:")
        for issue in bib_result["validation_issues"]:
            print(f"Reference {issue['reference_index'] + 1}:")
            for problem in issue["issues"]:
                print(f"  - {problem}")
    
    if "lookup_results" in bib_result:
        result = bib_result["lookup_results"]
        print("\nReference Lookup:")
        print(f"Title: {result.get('title', 'Unknown')}")
        print(f"Authors: {', '.join(result.get('authors', []))}")
        print(f"Year: {result.get('year', 'N/A')}")
        print(f"Venue: {result.get('venue', 'N/A')}")
        print(f"DOI: {result.get('doi', 'N/A')}")
        
        if "citations" in result:
            print("\nFormatted Citations:")
            for style, citation in result["citations"].items():
                print(f"\n{style.upper()}:")
                print(citation)

def print_research_gaps(gaps_result):
    """Print research gap analysis results in a formatted way."""
    print("\n🔎 RESEARCH GAP ANALYSIS 🔎")
    print("=" * 80)
    
    if "topic_overview" in gaps_result:
        print("Topic Overview:")
        print(gaps_result["topic_overview"])
    
    if "papers_analyzed" in gaps_result:
        print("\nPapers Analyzed:")
        for i, paper in enumerate(gaps_result["papers_analyzed"]):
            print(f"{i+1}. {paper.get('title', 'Unknown')}")
            print(f"   Authors: {paper.get('authors', 'Unknown')}")
            print(f"   Year: {paper.get('year', 'N/A')} | Venue: {paper.get('venue', 'N/A')}")
            if i < len(gaps_result["papers_analyzed"]) - 1:
                print("-" * 40)
    
    if "identified_gaps" in gaps_result:
        print("\nIdentified Research Gaps:")
        for i, gap in enumerate(gaps_result["identified_gaps"]):
            print(f"{i+1}. {gap.get('gap', 'Unknown gap')}")
            if gap.get('supporting_evidence'):
                print(f"   Evidence: {gap['supporting_evidence']}")
            if gap.get('potential_impact'):
                print(f"   Impact: {gap['potential_impact']}")
            if i < len(gaps_result["identified_gaps"]) - 1:
                print("-" * 40)
    
    if "future_directions" in gaps_result:
        print("\nFuture Research Directions:")
        for i, direction in enumerate(gaps_result["future_directions"]):
            print(f"{i+1}. {direction.get('direction', 'Unknown direction')}")
            if direction.get('relevance'):
                print(f"   Relevance: {direction['relevance']}")
            if direction.get('challenges'):
                print(f"   Challenges: {direction['challenges']}")
            if i < len(gaps_result["future_directions"]) - 1:
                print("-" * 40)
    
    if "methodology_suggestions" in gaps_result:
        print("\nMethodology Suggestions:")
        for i, suggestion in enumerate(gaps_result["methodology_suggestions"]):
            print(f"{i+1}. {suggestion}")

async def run_client():
    """Run the academic research assistant client."""
    print("🎓 Academic Research Assistant 🎓")
    print("=" * 80)
    print("Welcome! This assistant helps you with academic research tasks.")
    print("You can ask for paper searches, citation analysis, gap identification, and more.")
    print("Type 'exit' to quit.")
    print("=" * 80)
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get available tools from the MCP server
            mcp_tools = await session.list_tools()
            print(f"Connected to MCP server with {len(mcp_tools.tools)} research tools.")
            
            # Configure Gemini's function calling with our tools
            tools = [
                types.Tool(
                    function_declarations=[
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                k: v
                                for k, v in tool.inputSchema.items()
                                if k not in ["additionalProperties", "$schema"]
                            },
                        }
                    ]
                )
                for tool in mcp_tools.tools
            ]
            
            # Main interaction loop
            while True:
                try:
                    # Get user query
                    user_query = input("\n🔍 What would you like to research? ")
                    
                    if user_query.lower() in ["exit", "quit", "bye"]:
                        print("Thank you for using Academic Research Assistant. Goodbye!")
                        break
                    
                    print("Processing your request...")
                    
                    # Send query to Gemini for function calling
                    response = client.models.generate_content(
                        model="gemini-2.5-pro-exp-03-25",
                        contents=user_query,
                        config=types.GenerateContentConfig(
                            temperature=0,
                            tools=tools,
                        ),
                    )
                    
                    # Check if Gemini wants to call a function
                    if response.candidates[0].content.parts[0].function_call:
                        function_call = response.candidates[0].content.parts[0].function_call
                        
                        print(f"Tool: {function_call.name}")
                        
                        # Call the MCP tool
                        tool_result = await session.call_tool(
                            function_call.name, arguments=dict(function_call.args)
                        )
                        
                        # Parse the result
                        try:
                            result_content = tool_result.content[0].text
                            result_data = json.loads(result_content)
                            
                            # Display results based on the tool used
                            if function_call.name == "search_papers":
                                print_papers(result_data.get("papers", []))
                                
                            elif function_call.name == "analyze_citations":
                                print_citation_analysis(result_data)
                                
                            elif function_call.name == "analyze_paper":
                                print_paper_analysis(result_data)
                                
                            elif function_call.name == "manage_bibliography":
                                print_bibliography(result_data)
                                
                            elif function_call.name == "identify_research_gaps":
                                print_research_gaps(result_data)
                                
                            else:
                                print(json.dumps(result_data, indent=2))
                            
                        except json.JSONDecodeError:
                            print("Error parsing result:")
                            print(result_content)
                        except (IndexError, AttributeError) as e:
                            print(f"Error accessing result: {e}")
                            print(tool_result)
                    else:
                        # Gemini didn't call a function, just show its response
                        print("\nResponse:")
                        print(response.text)
                        
                except Exception as e:
                    print(f"Error: {e}")
                    
                print("\nPress Enter to continue...")
                input()

if __name__ == "__main__":
    asyncio.run(run_client())
