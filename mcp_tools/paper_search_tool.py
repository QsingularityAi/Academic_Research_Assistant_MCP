"""
Paper Search Tool: Searches multiple academic databases for papers matching criteria.
"""
import json
import asyncio
from typing import Optional, List, Dict, Any
from mcp.tool import Tool, AsyncTool
from scholarly import scholarly
import aiohttp
from semantic_scholar_api import SemanticScholarApi

class PaperSearchTool(AsyncTool):
    """Tool for searching academic papers across multiple databases."""
    
    def __init__(self, semantic_scholar_api_key: Optional[str] = None):
        """Initialize with optional API keys."""
        super().__init__()
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.ss_client = SemanticScholarApi(api_key=semantic_scholar_api_key)
        
    @property
    def name(self) -> str:
        return "search_papers"
    
    @property
    def description(self) -> str:
        return "Search for academic papers across Google Scholar, Semantic Scholar, and arXiv based on query parameters."
    
    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for finding papers"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)",
                    "default": 10
                },
                "start_year": {
                    "type": "integer",
                    "description": "Filter papers published on or after this year"
                },
                "end_year": {
                    "type": "integer",
                    "description": "Filter papers published on or before this year"
                },
                "sort_by": {
                    "type": "string",
                    "description": "How to sort the results",
                    "enum": ["relevance", "date"],
                    "default": "relevance"
                },
                "sources": {
                    "type": "array",
                    "description": "List of sources to search (default: all)",
                    "items": {
                        "type": "string",
                        "enum": ["google_scholar", "semantic_scholar", "arxiv"]
                    },
                    "default": ["google_scholar", "semantic_scholar", "arxiv"]
                }
            }
        }
    
    @property
    def outputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "papers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "authors": {"type": "array", "items": {"type": "string"}},
                            "year": {"type": "integer"},
                            "abstract": {"type": "string"},
                            "citation_count": {"type": "integer"},
                            "url": {"type": "string"},
                            "doi": {"type": "string"},
                            "source": {"type": "string"},
                            "venue": {"type": "string"}
                        }
                    }
                },
                "total_found": {"type": "integer"},
                "sources_searched": {"type": "array", "items": {"type": "string"}}
            }
        }
    
    async def _search_google_scholar(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Google Scholar for papers."""
        results = []
        try:
            search_query = scholarly.search_pubs(query)
            count = 0
            
            for i in range(min(max_results, 10)):  # Google Scholar has limitations
                try:
                    publication = next(search_query)
                    # Format the publication data
                    paper = {
                        "title": publication.get("bib", {}).get("title", "Unknown Title"),
                        "authors": publication.get("bib", {}).get("author", []),
                        "year": publication.get("bib", {}).get("pub_year"),
                        "abstract": publication.get("bib", {}).get("abstract", ""),
                        "citation_count": publication.get("num_citations", 0),
                        "url": publication.get("pub_url", ""),
                        "doi": None,  # Google Scholar doesn't provide DOI directly
                        "source": "google_scholar",
                        "venue": publication.get("bib", {}).get("venue", "")
                    }
                    results.append(paper)
                    count += 1
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Error processing Google Scholar result: {e}")
                    continue
        except Exception as e:
            print(f"Error searching Google Scholar: {e}")
        
        return results
    
    async def _search_semantic_scholar(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Semantic Scholar for papers."""
        results = []
        try:
            ss_results = self.ss_client.search_paper(query, limit=max_results, fields=[
                "title", "authors", "year", "abstract", "citationCount", "externalIds", "venue"
            ])
            
            for paper_data in ss_results.get("data", []):
                authors = [author.get("name", "") for author in paper_data.get("authors", [])]
                paper = {
                    "title": paper_data.get("title", "Unknown Title"),
                    "authors": authors,
                    "year": paper_data.get("year"),
                    "abstract": paper_data.get("abstract", ""),
                    "citation_count": paper_data.get("citationCount", 0),
                    "url": f"https://www.semanticscholar.org/paper/{paper_data.get('paperId')}" if paper_data.get('paperId') else "",
                    "doi": paper_data.get("externalIds", {}).get("DOI"),
                    "source": "semantic_scholar",
                    "venue": paper_data.get("venue", "")
                }
                results.append(paper)
        except Exception as e:
            print(f"Error searching Semantic Scholar: {e}")
        
        return results
    
    async def _search_arxiv(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search arXiv for papers."""
        results = []
        try:
            # Construct arXiv API query
            base_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": query,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    if response.status == 200:
                        # arXiv returns XML, we'll need to parse it
                        content = await response.text()
                        
                        # Simple parsing of XML (in production, use proper XML parsing)
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(content)
                        
                        # Define namespace
                        ns = {'atom': 'http://www.w3.org/2005/Atom',
                              'arxiv': 'http://arxiv.org/schemas/atom'}
                        
                        for entry in root.findall('.//atom:entry', ns):
                            title = entry.find('./atom:title', ns).text.strip()
                            
                            # Extract authors
                            authors = []
                            for author in entry.findall('./atom:author/atom:name', ns):
                                authors.append(author.text.strip())
                            
                            # Get abstract
                            abstract = entry.find('./atom:summary', ns).text.strip()
                            
                            # Get URL and ID
                            url = entry.find('./atom:id', ns).text.strip()
                            
                            # Get published date (format: 2022-01-01T00:00:00Z)
                            published = entry.find('./atom:published', ns).text.strip()
                            year = int(published.split('-')[0]) if published else None
                            
                            # Create paper object
                            paper = {
                                "title": title,
                                "authors": authors,
                                "year": year,
                                "abstract": abstract,
                                "citation_count": None,  # arXiv doesn't provide this
                                "url": url,
                                "doi": None,  # Extract from links if available
                                "source": "arxiv",
                                "venue": "arXiv"  # All papers here are from arXiv
                            }
                            
                            # Extract DOI if available
                            for link in entry.findall('./atom:link', ns):
                                if link.get('title') == 'doi':
                                    paper["doi"] = link.get('href').split('doi.org/')[-1]
                            
                            results.append(paper)
        except Exception as e:
            print(f"Error searching arXiv: {e}")
        
        return results

    async def invoke(self, query: str, max_results: int = 10, 
                     start_year: Optional[int] = None, 
                     end_year: Optional[int] = None,
                     sort_by: str = "relevance",
                     sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search for academic papers across multiple databases.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            start_year: Filter papers published on or after this year
            end_year: Filter papers published on or before this year
            sort_by: Sort method (relevance or date)
            sources: List of sources to search
            
        Returns:
            Dict containing search results
        """
        if sources is None:
            sources = ["google_scholar", "semantic_scholar", "arxiv"]
        
        all_results = []
        sources_searched = []
        
        # Run searches in parallel
        tasks = []
        if "google_scholar" in sources:
            tasks.append(self._search_google_scholar(query, max_results))
            sources_searched.append("google_scholar")
        
        if "semantic_scholar" in sources:
            tasks.append(self._search_semantic_scholar(query, max_results))
            sources_searched.append("semantic_scholar")
        
        if "arxiv" in sources:
            tasks.append(self._search_arxiv(query, max_results))
            sources_searched.append("arxiv")
        
        # Gather all results
        search_results = await asyncio.gather(*tasks)
        for results in search_results:
            all_results.extend(results)
            
        # Apply filters
        if start_year:
            all_results = [paper for paper in all_results if paper.get("year") and paper.get("year") >= start_year]
        
        if end_year:
            all_results = [paper for paper in all_results if paper.get("year") and paper.get("year") <= end_year]
        
        # Sort results
        if sort_by == "date":
            all_results.sort(key=lambda x: x.get("year", 0) or 0, reverse=True)
        else:  # relevance - we'll use citation count as a proxy for relevance
            all_results.sort(key=lambda x: x.get("citation_count", 0) or 0, reverse=True)
        
        # Limit results
        all_results = all_results[:max_results]
        
        return {
            "papers": all_results,
            "total_found": len(all_results),
            "sources_searched": sources_searched
        }
