"""
Citation Analysis Tool: Generates and analyzes citation networks for academic papers.
"""
import json
import asyncio
from typing import Optional, List, Dict, Any, Set, Tuple
from mcp.tool import Tool, AsyncTool
import networkx as nx
from semantic_scholar_api import SemanticScholarApi
import base64

class CitationAnalysisTool(AsyncTool):
    """Tool for analyzing citation networks among academic papers."""
    
    def __init__(self, semantic_scholar_api_key: Optional[str] = None):
        """Initialize with optional API keys."""
        super().__init__()
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.ss_client = SemanticScholarApi(api_key=semantic_scholar_api_key)
        
    @property
    def name(self) -> str:
        return "analyze_citations"
    
    @property
    def description(self) -> str:
        return "Analyze citation networks for academic papers, identifying key papers and research clusters."
    
    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["paper_id"],
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "Semantic Scholar Paper ID, DOI, or arXiv ID"
                },
                "depth": {
                    "type": "integer",
                    "description": "Depth of citation network to explore (1-3)",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 3
                },
                "direction": {
                    "type": "string",
                    "description": "Direction of citations to explore",
                    "enum": ["citing", "cited_by", "both"],
                    "default": "both"
                },
                "max_papers": {
                    "type": "integer",
                    "description": "Maximum number of papers to include in the network",
                    "default": 50,
                    "maximum": 100
                }
            }
        }
    
    @property
    def outputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "network_stats": {
                    "type": "object",
                    "properties": {
                        "nodes": {"type": "integer"},
                        "edges": {"type": "integer"},
                        "average_citations": {"type": "number"},
                        "most_cited_papers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "authors": {"type": "string"},
                                    "year": {"type": "integer"},
                                    "citation_count": {"type": "integer"}
                                }
                            }
                        }
                    }
                },
                "clusters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "size": {"type": "integer"},
                            "key_papers": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    }
                },
                "visualization": {
                    "type": "string",
                    "description": "HTML visualization of the citation network"
                }
            }
        }
    
    async def _get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get paper details from Semantic Scholar."""
        try:
            # Try to identify the type of ID
            if paper_id.startswith("10."):  # Likely a DOI
                search_id = f"DOI:{paper_id}"
            elif paper_id.startswith("arXiv:"):
                search_id = paper_id
            elif "arxiv" in paper_id.lower():
                arxiv_id = paper_id.split("/")[-1]
                search_id = f"arXiv:{arxiv_id}"
            else:  # Assume it's a Semantic Scholar ID
                search_id = paper_id
                
            paper = self.ss_client.get_paper(search_id, fields=[
                "paperId", "title", "authors", "year", "citationCount", 
                "citations", "references", "abstract", "venue"
            ])
            
            return paper
        except Exception as e:
            print(f"Error fetching paper details: {e}")
            return {}
    
    async def _build_citation_network(self, paper_id: str, depth: int, 
                                     direction: str, max_papers: int) -> Tuple[nx.DiGraph, Dict[str, Dict]]:
        """
        Build a citation network around the specified paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            depth: Depth of network to explore
            direction: Direction of citations to explore
            max_papers: Maximum papers to include
            
        Returns:
            Tuple of (network graph, paper details dictionary)
        """
        G = nx.DiGraph()
        paper_details = {}
        papers_to_process = {paper_id}
        processed_papers = set()
        
        # Process papers up to specified depth
        for current_depth in range(depth + 1):
            if not papers_to_process or len(paper_details) >= max_papers:
                break
                
            next_level_papers = set()
            
            # Process all papers at the current level
            for current_paper_id in papers_to_process:
                if current_paper_id in processed_papers or len(paper_details) >= max_papers:
                    continue
                    
                # Get paper details
                paper_data = await self._get_paper_details(current_paper_id)
                if not paper_data:
                    continue
                    
                # Store paper details
                paper_details[current_paper_id] = {
                    "id": current_paper_id,
                    "title": paper_data.get("title", "Unknown Title"),
                    "authors": ", ".join([a.get("name", "") for a in paper_data.get("authors", [])]),
                    "year": paper_data.get("year"),
                    "citation_count": paper_data.get("citationCount", 0),
                    "abstract": paper_data.get("abstract", ""),
                    "venue": paper_data.get("venue", "")
                }
                
                # Add node to graph
                G.add_node(current_paper_id, **paper_details[current_paper_id])
                
                # Add citing papers (papers that cite this paper)
                if direction in ["cited_by", "both"] and current_depth < depth:
                    citations = paper_data.get("citations", [])
                    for citation in citations[:min(20, len(citations))]:  # Limit to 20 citations per paper
                        citing_id = citation.get("paperId")
                        if citing_id and len(paper_details) < max_papers:
                            G.add_edge(citing_id, current_paper_id)  # Direction: citing -> cited
                            next_level_papers.add(citing_id)
                
                # Add cited papers (papers that this paper cites)
                if direction in ["citing", "both"] and current_depth < depth:
                    references = paper_data.get("references", [])
                    for reference in references[:min(20, len(references))]:  # Limit to 20 references per paper
                        cited_id = reference.get("paperId")
                        if cited_id and len(paper_details) < max_papers:
                            G.add_edge(current_paper_id, cited_id)  # Direction: citing -> cited
                            next_level_papers.add(cited_id)
                
                processed_papers.add(current_paper_id)
            
            # Update papers to process for next level
            papers_to_process = next_level_papers - processed_papers
        
        return G, paper_details
    
    def _identify_clusters(self, G: nx.DiGraph) -> List[Dict[str, Any]]:
        """Identify research clusters in the citation network."""
        # Convert directed graph to undirected for community detection
        G_undirected = G.to_undirected()
        
        # Use community detection algorithm
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G_undirected)
        except ImportError:
            # Fallback to connected components if community module not available
            clusters = list(nx.connected_components(G_undirected))
            partition = {}
            for i, cluster in enumerate(clusters):
                for node in cluster:
                    partition[node] = i
        
        # Group nodes by community
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        
        # Create cluster output
        clusters = []
        for community_id, nodes in communities.items():
            if len(nodes) < 2:  # Skip singleton clusters
                continue
                
            # Identify key papers using PageRank
            subgraph = G.subgraph(nodes)
            pageranks = nx.pagerank(subgraph)
            key_papers = sorted(pageranks.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Get primary paper title for naming the cluster
            top_paper_id = key_papers[0][0] if key_papers else nodes[0]
            primary_title = G.nodes[top_paper_id].get("title", "Unknown Cluster")
            
            # Create cluster info
            cluster = {
                "name": f"Cluster {community_id + 1}: {primary_title[:50]}...",
                "size": len(nodes),
                "key_papers": [G.nodes[paper_id].get("title", "Unknown") for paper_id, _ in key_papers]
            }
            clusters.append(cluster)
        
        return sorted(clusters, key=lambda x: x["size"], reverse=True)
    
    def _create_network_visualization(self, G: nx.DiGraph, paper_details: Dict[str, Dict]) -> str:
        """Create an HTML visualization of the citation network."""
        try:
            from pyvis.network import Network
            
            # Create network visualization
            net = Network(height="600px", width="100%", directed=True, notebook=False)
            
            # Calculate node sizes based on citation count (normalized)
            citation_counts = [paper_details[node].get("citation_count", 0) for node in G.nodes() if node in paper_details]
            max_citations = max(citation_counts) if citation_counts else 1
            min_size, max_size = 10, 50
            
            # Add nodes with attributes
            for node in G.nodes():
                if node in paper_details:
                    paper = paper_details[node]
                    # Calculate node size
                    size = min_size + (paper.get("citation_count", 0) / max(1, max_citations)) * (max_size - min_size)
                    
                    # Create label
                    label = f"{paper.get('title', 'Unknown')} ({paper.get('year', 'N/A')})"
                    
                    # Create hover tooltip
                    title = f"<strong>{paper.get('title', 'Unknown')}</strong><br>" \
                            f"<em>{paper.get('authors', '')}</em><br>" \
                            f"Year: {paper.get('year', 'N/A')}<br>" \
                            f"Citations: {paper.get('citation_count', 0)}<br>"
                    
                    # Add node to network
                    net.add_node(
                        node, 
                        label=label[:30] + "..." if len(label) > 30 else label, 
                        title=title,
                        size=size,
                        color="#97c2fc"  # Default blue color
                    )
            
            # Add edges
            for source, target in G.edges():
                net.add_edge(source, target, arrows="to")
            
            # Generate and return HTML
            html_path = "/tmp/citation_network.html"
            net.save_graph(html_path)
            
            # Read HTML content
            with open(html_path, "r") as f:
                html_content = f.read()
                
            return html_content
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return "<p>Error creating visualization. Please try again with fewer papers.</p>"

    async def invoke(self, paper_id: str, depth: int = 1, 
                     direction: str = "both", max_papers: int = 50) -> Dict[str, Any]:
        """
        Analyze citation networks for academic papers.
        
        Args:
            paper_id: Semantic Scholar Paper ID, DOI, or arXiv ID
            depth: Depth of citation network to explore
            direction: Direction of citations to explore
            max_papers: Maximum number of papers in the network
            
        Returns:
            Dict containing citation analysis results
        """
        # Build citation network
        G, paper_details = await self._build_citation_network(
            paper_id, depth, direction, max_papers
        )
        
        # Calculate network statistics
        if len(G.nodes()) == 0:
            return {
                "error": "Could not build citation network. Please check the paper ID."
            }
        
        # Calculate most cited papers
        in_degrees = dict(G.in_degree())
        most_cited = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        most_cited_papers = []
        for paper_id, citations in most_cited:
            if paper_id in paper_details:
                paper = paper_details[paper_id]
                most_cited_papers.append({
                    "title": paper.get("title", "Unknown"),
                    "authors": paper.get("authors", ""),
                    "year": paper.get("year"),
                    "citation_count": citations
                })
        
        # Calculate average citations
        avg_citations = sum(in_degrees.values()) / max(1, len(in_degrees))
        
        # Identify research clusters
        clusters = self._identify_clusters(G)
        
        # Create network visualization
        visualization = self._create_network_visualization(G, paper_details)
        
        # Prepare response
        response = {
            "network_stats": {
                "nodes": len(G.nodes()),
                "edges": len(G.edges()),
                "average_citations": round(avg_citations, 2),
                "most_cited_papers": most_cited_papers
            },
            "clusters": clusters,
            "visualization": visualization
        }
        
        return response
