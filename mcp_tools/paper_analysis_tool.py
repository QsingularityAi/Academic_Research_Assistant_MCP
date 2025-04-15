"""
Paper Analysis Tool: Analyzes academic papers to extract key findings and summaries.
"""
import json
import asyncio
from typing import Optional, List, Dict, Any
from mcp.tool import Tool, AsyncTool
import aiohttp
from semantic_scholar_api import SemanticScholarApi
import os
from google import genai

class PaperAnalysisTool(AsyncTool):
    """Tool for analyzing academic papers to extract key information."""
    
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
        return "analyze_paper"
    
    @property
    def description(self) -> str:
        return "Analyze academic papers to extract summaries, key findings, methodologies, and other important content."
    
    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "Semantic Scholar Paper ID, DOI, or arXiv ID"
                },
                "paper_url": {
                    "type": "string",
                    "description": "URL to the paper (PDF or HTML)"
                },
                "paper_text": {
                    "type": "string",
                    "description": "Full text of the paper or abstract if full text not available"
                },
                "analysis_type": {
                    "type": "string",
                    "description": "Type of analysis to perform",
                    "enum": ["summary", "key_findings", "methodology", "gaps", "full"],
                    "default": "full"
                }
            },
            "oneOf": [
                {"required": ["paper_id"]},
                {"required": ["paper_url"]},
                {"required": ["paper_text"]}
            ]
        }
    
    @property
    def outputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "paper_info": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "authors": {"type": "array", "items": {"type": "string"}},
                        "year": {"type": "integer"},
                        "venue": {"type": "string"},
                        "abstract": {"type": "string"}
                    }
                },
                "summary": {"type": "string"},
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "methodology": {"type": "string"},
                "research_gaps": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "future_directions": {
                    "type": "array",
                    "items": {"type": "string"}
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
                "paperId", "title", "authors", "year", "abstract", "venue", "url"
            ])
            
            return paper
        except Exception as e:
            print(f"Error fetching paper details: {e}")
            return {}
    
    async def _fetch_paper_text(self, url: str) -> str:
        """Fetch paper text from URL."""
        try:
            # For simplicity, we'll just get the abstract if it's a PDF
            # In a production system, you'd use a PDF extraction library
            if url.endswith(".pdf"):
                # Get paper ID from Semantic Scholar
                search_results = self.ss_client.search_paper(url)
                if search_results and search_results.get("data"):
                    paper_id = search_results["data"][0].get("paperId")
                    paper = await self._get_paper_details(paper_id)
                    return paper.get("abstract", "")
                return "Could not extract text from PDF. Please provide paper ID or text directly."
            
            # If it's HTML, fetch the page and extract text
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Simple HTML text extraction
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.extract()
                        
                        # Get text
                        text = soup.get_text()
                        
                        # Break into lines and remove leading/trailing space
                        lines = (line.strip() for line in text.splitlines())
                        # Break multi-headlines into a line each
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        # Remove blank lines
                        text = '\n'.join(chunk for chunk in chunks if chunk)
                        
                        return text
                    
                    return f"Error fetching URL: HTTP {response.status}"
        except Exception as e:
            print(f"Error fetching paper text: {e}")
            return f"Error fetching paper text: {str(e)}"
    
    async def _analyze_with_gemini(self, paper_text: str, paper_info: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """Use Gemini to analyze paper content."""
        if not self.model:
            return {
                "error": "Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
            }
        
        # Prepare context for Gemini
        title = paper_info.get("title", "")
        authors = ", ".join([a.get("name", "") for a in paper_info.get("authors", [])])
        year = paper_info.get("year", "")
        abstract = paper_info.get("abstract", "")
        
        # Prepare the paper context
        paper_context = f"Title: {title}\nAuthors: {authors}\nYear: {year}\nAbstract: {abstract}\n\nFull Text:\n{paper_text}"
        
        # Create analysis prompts based on analysis type
        prompts = {
            "summary": f"Please provide a comprehensive summary of the following academic paper. Capture the main points, contributions, and conclusions in about 3-5 paragraphs.\n\n{paper_context}",
            
            "key_findings": f"Extract and list the key findings, contributions, and results from the following academic paper. Provide each finding as a separate point.\n\n{paper_context}",
            
            "methodology": f"Describe in detail the methodology used in the following academic paper. Include information about data collection, experimental setup, analytical approaches, and any limitations mentioned.\n\n{paper_context}",
            
            "gaps": f"Identify any research gaps, limitations, or areas for future work mentioned in the following academic paper. Also suggest potential future research directions based on this paper.\n\n{paper_context}",
            
            "full": f"Provide a comprehensive analysis of the following academic paper with these sections:\n1. Summary (3-5 paragraphs)\n2. Key Findings (bullet points)\n3. Methodology\n4. Research Gaps and Limitations\n5. Future Research Directions\n\n{paper_context}"
        }
        
        try:
            # Get the appropriate prompt
            prompt = prompts.get(analysis_type, prompts["full"])
            
            # Generate content with Gemini
            response = self.model.text(prompt)
            analysis = response.result
            
            # Process the response based on analysis type
            if analysis_type == "summary":
                return {"summary": analysis}
            
            elif analysis_type == "key_findings":
                # Extract bullet points or numbered list items
                findings = []
                for line in analysis.split("\n"):
                    line = line.strip()
                    if line and (line.startswith("- ") or line.startswith("* ") or 
                                 (line[0].isdigit() and line[1:].startswith(". "))):
                        findings.append(line.lstrip("- *0123456789. "))
                
                return {"key_findings": findings}
            
            elif analysis_type == "methodology":
                return {"methodology": analysis}
            
            elif analysis_type == "gaps":
                # Extract gaps and future directions
                sections = analysis.split("\n\n")
                gaps = []
                future = []
                
                current_section = "gaps"
                for section in sections:
                    if "future" in section.lower():
                        current_section = "future"
                        
                    lines = section.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line and (line.startswith("- ") or line.startswith("* ") or 
                                    (line[0].isdigit() and line[1:].startswith(". "))):
                            if current_section == "gaps":
                                gaps.append(line.lstrip("- *0123456789. "))
                            else:
                                future.append(line.lstrip("- *0123456789. "))
                
                return {
                    "research_gaps": gaps,
                    "future_directions": future
                }
            
            elif analysis_type == "full":
                # Parse the full analysis into sections
                sections = analysis.split("\n\n")
                
                # Initialize result structure
                result = {
                    "summary": "",
                    "key_findings": [],
                    "methodology": "",
                    "research_gaps": [],
                    "future_directions": []
                }
                
                current_section = None
                for section in sections:
                    lower_section = section.lower()
                    
                    if "summary" in lower_section or "overview" in lower_section:
                        current_section = "summary"
                        result["summary"] = section.split("\n", 1)[1] if "\n" in section else ""
                        
                    elif "key finding" in lower_section or "result" in lower_section or "contribution" in lower_section:
                        current_section = "key_findings"
                        for line in section.split("\n"):
                            line = line.strip()
                            if line and (line.startswith("- ") or line.startswith("* ") or 
                                        (line[0].isdigit() and line[1:].startswith(". "))):
                                result["key_findings"].append(line.lstrip("- *0123456789. "))
                        
                    elif "methodology" in lower_section or "method" in lower_section:
                        current_section = "methodology"
                        result["methodology"] = section.split("\n", 1)[1] if "\n" in section else section
                        
                    elif "gap" in lower_section or "limitation" in lower_section:
                        current_section = "research_gaps"
                        for line in section.split("\n"):
                            line = line.strip()
                            if line and (line.startswith("- ") or line.startswith("* ") or 
                                        (line[0].isdigit() and line[1:].startswith(". "))):
                                result["research_gaps"].append(line.lstrip("- *0123456789. "))
                        
                    elif "future" in lower_section:
                        current_section = "future_directions"
                        for line in section.split("\n"):
                            line = line.strip()
                            if line and (line.startswith("- ") or line.startswith("* ") or 
                                        (line[0].isdigit() and line[1:].startswith(". "))):
                                result["future_directions"].append(line.lstrip("- *0123456789. "))
                
                return result
            
            return {"error": "Unknown analysis type"}
            
        except Exception as e:
            print(f"Error analyzing with Gemini: {e}")
            return {"error": f"Error analyzing with Gemini: {str(e)}"}

    async def invoke(self, paper_id: Optional[str] = None, paper_url: Optional[str] = None, 
                     paper_text: Optional[str] = None, analysis_type: str = "full") -> Dict[str, Any]:
        """
        Analyze an academic paper to extract key information.
        
        Args:
            paper_id: Semantic Scholar Paper ID, DOI, or arXiv ID
            paper_url: URL to the paper
            paper_text: Full text of the paper
            analysis_type: Type of analysis to perform
            
        Returns:
            Dict containing analysis results
        """
        paper_info = {}
        text_to_analyze = ""
        
        # Get paper information based on provided inputs
        if paper_id:
            paper_info = await self._get_paper_details(paper_id)
            text_to_analyze = paper_info.get("abstract", "")
        
        elif paper_url:
            # Try to extract text from URL
            text_to_analyze = await self._fetch_paper_text(paper_url)
            
            # Try to get paper metadata
            search_results = self.ss_client.search_paper(paper_url)
            if search_results and search_results.get("data"):
                paper_id = search_results["data"][0].get("paperId")
                paper_info = await self._get_paper_details(paper_id)
        
        elif paper_text:
            text_to_analyze = paper_text
            
            # Try to identify the paper based on text
            # This is a simplification - in a real system you'd use more sophisticated methods
            first_paragraph = paper_text.split("\n\n")[0]
            search_results = self.ss_client.search_paper(first_paragraph[:100])
            if search_results and search_results.get("data"):
                paper_id = search_results["data"][0].get("paperId")
                paper_info = await self._get_paper_details(paper_id)
        
        else:
            return {"error": "One of paper_id, paper_url, or paper_text must be provided"}
        
        # If we have no text to analyze, return an error
        if not text_to_analyze:
            return {"error": "Could not extract text to analyze from the provided inputs"}
        
        # Perform analysis with Gemini
        analysis_results = await self._analyze_with_gemini(
            text_to_analyze, paper_info, analysis_type
        )
        
        # Prepare response
        authors_formatted = []
        if paper_info and paper_info.get("authors"):
            authors_formatted = [author.get("name", "") for author in paper_info.get("authors", [])]
        
        response = {
            "paper_info": {
                "title": paper_info.get("title", "Unknown Title"),
                "authors": authors_formatted,
                "year": paper_info.get("year"),
                "venue": paper_info.get("venue", ""),
                "abstract": paper_info.get("abstract", "")
            }
        }
        
        # Add analysis results
        response.update(analysis_results)
        
        return response
