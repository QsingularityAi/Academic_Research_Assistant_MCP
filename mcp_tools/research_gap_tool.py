"""
Research Gap Tool: Identifies research gaps in academic literature.
"""
import json
import asyncio
from typing import Optional, List, Dict, Any, Set
from mcp.tool import Tool, AsyncTool
import os
from google import genai
from semantic_scholar_api import SemanticScholarApi

class ResearchGapTool(AsyncTool):
    """Tool for identifying research gaps in academic literature."""
    
    def __init__(self, gemini_api_key: Optional[str] = None, semantic_scholar_api_key: Optional[str] = None):
        """Initialize with optional API keys."""
        super().__init__()
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.ss_client = SemanticScholarApi(api_key=semantic_scholar_api_key)
        
        # Initialize Gemini client if API key available
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.models.get_model("gemini-2.5-pro-exp-03-25")
        else:
            self.model = None
        
    @property
    def name(self) -> str:
        return "identify_research_gaps"
    
    @property
    def description(self) -> str:
        return "Identify research gaps and future directions in academic literature based on a research topic or field."
    
    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["topic"],
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Research topic or field to analyze"
                },
                "papers": {
                    "type": "array",
                    "description": "List of paper IDs (DOIs, arXiv IDs, or Semantic Scholar IDs) to analyze",
                    "items": {"type": "string"}
                },
                "depth": {
                    "type": "integer",
                    "description": "Depth of analysis (1-3)",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 3
                },
                "num_papers": {
                    "type": "integer",
                    "description": "Number of papers to analyze if none provided",
                    "default": 10,
                    "maximum": 20
                }
            }
        }
    
    @property
    def outputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic_overview": {"type": "string"},
                "papers_analyzed": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "authors": {"type": "string"},
                            "year": {"type": "integer"},
                            "venue": {"type": "string"}
                        }
                    }
                },
                "identified_gaps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "gap": {"type": "string"},
                            "supporting_evidence": {"type": "string"},
                            "potential_impact": {"type": "string"}
                        }
                    }
                },
                "future_directions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "direction": {"type": "string"},
                            "relevance": {"type": "string"},
                            "challenges": {"type": "string"}
                        }
                    }
                },
                "methodology_suggestions": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    
    async def _search_papers(self, topic: str, num_papers: int) -> List[Dict[str, Any]]:
        """Search for papers on the given topic."""
        try:
            search_results = self.ss_client.search_paper(
                topic, 
                limit=num_papers, 
                fields=["paperId", "title", "authors", "year", "abstract", "venue", "citationCount"]
            )
            
            if not search_results or not search_results.get("data"):
                return []
                
            # Sort by citation count to get more influential papers
            papers = search_results["data"]
            papers.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
            
            return papers[:num_papers]
        except Exception as e:
            print(f"Error searching papers: {e}")
            return []
    
    async def _get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed information about a paper."""
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
                "paperId", "title", "authors", "year", "abstract", "venue", 
                "citationCount", "citations", "references", "tldr"
            ])
            
            return paper
        except Exception as e:
            print(f"Error fetching paper details: {e}")
            return {}
    
    async def _collect_papers(self, topic: str, provided_papers: Optional[List[str]], 
                             num_papers: int) -> List[Dict[str, Any]]:
        """Collect papers to analyze from provided IDs or by search."""
        papers = []
        
        # If papers are provided, get their details
        if provided_papers and len(provided_papers) > 0:
            for paper_id in provided_papers:
                paper = await self._get_paper_details(paper_id)
                if paper:
                    papers.append(paper)
        
        # If we don't have enough papers, search for more
        if len(papers) < num_papers:
            search_results = await self._search_papers(topic, num_papers - len(papers))
            # Avoid duplicates
            existing_ids = {p.get("paperId") for p in papers if p.get("paperId")}
            for paper in search_results:
                if paper.get("paperId") not in existing_ids:
                    papers.append(paper)
                    existing_ids.add(paper.get("paperId"))
        
        return papers[:num_papers]
    
    async def _analyze_papers_with_gemini(self, topic: str, papers: List[Dict[str, Any]], 
                                          depth: int) -> Dict[str, Any]:
        """Use Gemini to analyze papers and identify research gaps."""
        if not self.model:
            return {
                "error": "Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
            }
        
        # Prepare context for Gemini
        papers_context = []
        for paper in papers:
            title = paper.get("title", "Unknown Title")
            authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])])
            year = paper.get("year", "Unknown Year")
            abstract = paper.get("abstract", "No abstract available")
            tldr = paper.get("tldr", {}).get("text", "")
            
            paper_context = f"Title: {title}\nAuthors: {authors}\nYear: {year}\nAbstract: {abstract}\nSummary: {tldr}\n\n"
            papers_context.append(paper_context)
        
        # Create detailed prompt for Gemini
        depth_descriptions = {
            1: "a basic analysis identifying major research gaps",
            2: "a moderate analysis identifying nuanced research gaps and methodology suggestions",
            3: "a comprehensive analysis with detailed research gaps, future directions, and methodology suggestions"
        }
        
        prompt = f"""As a research advisor, analyze the following papers on the topic of "{topic}" and provide {depth_descriptions.get(depth, 'an analysis')}. 

Papers to analyze:
{"".join(papers_context)}

Based on these papers, please provide:

1. A brief overview of the current state of research on {topic}
2. Identification of key research gaps in this field, including:
   - Description of the gap
   - Supporting evidence from the papers
   - Potential impact of addressing this gap
3. Suggested future research directions, including:
   - Description of potential research direction
   - Relevance to the current state of the field
   - Potential challenges or obstacles
4. Methodological suggestions for future research

Format your response with clear section headers and bullet points for each identified gap, direction, and suggestion.
"""
        
        try:
            # Generate content with Gemini
            response = self.model.text(prompt)
            analysis = response.result
            
            # Parse the response to extract structured information
            sections = analysis.split("\n\n")
            
            topic_overview = ""
            identified_gaps = []
            future_directions = []
            methodology_suggestions = []
            
            current_section = None
            current_item = {}
            
            for section in sections:
                lower_section = section.lower()
                
                if "overview" in lower_section or "state of research" in lower_section:
                    current_section = "overview"
                    topic_overview = section
                
                elif "gap" in lower_section or "limitation" in lower_section:
                    current_section = "gaps"
                    lines = section.split("\n")
                    
                    # Check if this is a new gap or continuing description
                    if lines[0].strip().startswith(("- ", "* ", "•")) or (lines[0].strip()[0].isdigit() and ":" in lines[0]):
                        # Save previous item if exists
                        if current_item and current_section == "gaps":
                            identified_gaps.append(current_item)
                            current_item = {}
                        
                        # Start new gap
                        gap_description = lines[0].strip().lstrip("- *•0123456789.:) ")
                        current_item = {"gap": gap_description, "supporting_evidence": "", "potential_impact": ""}
                    
                    # Add details to current gap
                    for line in lines[1:]:
                        line = line.strip()
                        if "evidence" in line.lower() or "support" in line.lower():
                            current_item["supporting_evidence"] = line.split(":", 1)[1].strip() if ":" in line else line
                        elif "impact" in line.lower() or "importance" in line.lower():
                            current_item["potential_impact"] = line.split(":", 1)[1].strip() if ":" in line else line
                    
                elif "future" in lower_section or "direction" in lower_section:
                    # Save previous item if exists
                    if current_item:
                        if current_section == "gaps":
                            identified_gaps.append(current_item)
                        elif current_section == "directions":
                            future_directions.append(current_item)
                        current_item = {}
                    
                    current_section = "directions"
                    lines = section.split("\n")
                    
                    # Check if this is a new direction or continuing description
                    if lines[0].strip().startswith(("- ", "* ", "•")) or (lines[0].strip()[0].isdigit() and ":" in lines[0]):
                        direction_description = lines[0].strip().lstrip("- *•0123456789.:) ")
                        current_item = {"direction": direction_description, "relevance": "", "challenges": ""}
                    
                    # Add details to current direction
                    for line in lines[1:]:
                        line = line.strip()
                        if "relevan" in line.lower():
                            current_item["relevance"] = line.split(":", 1)[1].strip() if ":" in line else line
                        elif "challenge" in line.lower() or "obstacle" in line.lower():
                            current_item["challenges"] = line.split(":", 1)[1].strip() if ":" in line else line
                
                elif "method" in lower_section or "suggestion" in lower_section:
                    # Save previous item if exists
                    if current_item:
                        if current_section == "gaps":
                            identified_gaps.append(current_item)
                        elif current_section == "directions":
                            future_directions.append(current_item)
                        current_item = {}
                    
                    current_section = "methods"
                    
                    # Extract methodology suggestions
                    lines = section.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line and (line.startswith(("- ", "* ", "•")) or (line[0].isdigit() and line[1:].startswith(". "))):
                            suggestion = line.lstrip("- *•0123456789.: ")
                            methodology_suggestions.append(suggestion)
            
            # Add the last item if it exists
            if current_item:
                if current_section == "gaps":
                    identified_gaps.append(current_item)
                elif current_section == "directions":
                    future_directions.append(current_item)
            
            # Clean up any empty entries
            identified_gaps = [g for g in identified_gaps if g.get("gap")]
            future_directions = [d for d in future_directions if d.get("direction")]
            
            # Create formatted output
            return {
                "topic_overview": topic_overview,
                "identified_gaps": identified_gaps,
                "future_directions": future_directions,
                "methodology_suggestions": methodology_suggestions
            }
            
        except Exception as e:
            print(f"Error analyzing with Gemini: {e}")
            return {"error": f"Error analyzing with Gemini: {str(e)}"}

    async def invoke(self, topic: str, papers: Optional[List[str]] = None, 
                     depth: int = 1, num_papers: int = 10) -> Dict[str, Any]:
        """
        Identify research gaps in academic literature.
        
        Args:
            topic: Research topic or field to analyze
            papers: List of paper IDs to analyze
            depth: Depth of analysis (1-3)
            num_papers: Number of papers to analyze if none provided
            
        Returns:
            Dict containing research gap analysis
        """
        # Collect papers to analyze
        collected_papers = await self._collect_papers(topic, papers, num_papers)
        
        if not collected_papers:
            return {"error": "No papers found to analyze. Please check your topic or provide paper IDs."}
        
        # Create formatted paper list for output
        papers_analyzed = []
        for paper in collected_papers:
            papers_analyzed.append({
                "title": paper.get("title", "Unknown Title"),
                "authors": ", ".join([a.get("name", "") for a in paper.get("authors", [])]),
                "year": paper.get("year"),
                "venue": paper.get("venue", "")
            })
        
        # Analyze papers with Gemini
        analysis_results = await self._analyze_papers_with_gemini(
            topic, collected_papers, depth
        )
        
        if "error" in analysis_results:
            return analysis_results
        
        # Prepare response
        response = {
            "papers_analyzed": papers_analyzed
        }
        
        # Add analysis results
        response.update(analysis_results)
        
        return response
