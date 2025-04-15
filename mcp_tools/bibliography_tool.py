"""
Bibliography Tool: Manages academic references and citations.
"""
import json
import asyncio
from typing import Optional, List, Dict, Any
from mcp.tool import Tool, AsyncTool
import re
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
from semantic_scholar_api import SemanticScholarApi

class BibliographyTool(AsyncTool):
    """Tool for managing academic references and citations."""
    
    def __init__(self, semantic_scholar_api_key: Optional[str] = None):
        """Initialize with optional API keys."""
        super().__init__()
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.ss_client = SemanticScholarApi(api_key=semantic_scholar_api_key)
        
    @property
    def name(self) -> str:
        return "manage_bibliography"
    
    @property
    def description(self) -> str:
        return "Manage academic references, format citations, and convert between citation formats."
    
    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform with the bibliography",
                    "enum": ["format", "convert", "lookup", "validate"],
                    "default": "format"
                },
                "references": {
                    "type": "array",
                    "description": "List of references to process",
                    "items": {"type": "string"}
                },
                "bibtex": {
                    "type": "string",
                    "description": "BibTeX format references as a single string"
                },
                "doi": {
                    "type": "string",
                    "description": "DOI to look up"
                },
                "title": {
                    "type": "string",
                    "description": "Paper title to look up"
                },
                "from_format": {
                    "type": "string",
                    "description": "Source citation format",
                    "enum": ["bibtex", "apa", "mla", "chicago", "ieee", "harvard"],
                    "default": "bibtex"
                },
                "to_format": {
                    "type": "string",
                    "description": "Target citation format",
                    "enum": ["bibtex", "apa", "mla", "chicago", "ieee", "harvard"],
                    "default": "apa"
                }
            },
            "oneOf": [
                {"required": ["references"]},
                {"required": ["bibtex"]},
                {"required": ["doi"]},
                {"required": ["title"]}
            ]
        }
    
    @property
    def outputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "formatted_references": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "bibtex": {
                    "type": "string",
                    "description": "BibTeX formatted references"
                },
                "validation_issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "reference_index": {"type": "integer"},
                            "issues": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "lookup_results": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "authors": {"type": "array", "items": {"type": "string"}},
                        "year": {"type": "integer"},
                        "venue": {"type": "string"},
                        "doi": {"type": "string"},
                        "citations": {
                            "type": "object",
                            "properties": {
                                "bibtex": {"type": "string"},
                                "apa": {"type": "string"},
                                "mla": {"type": "string"},
                                "chicago": {"type": "string"},
                                "ieee": {"type": "string"},
                                "harvard": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    
    def _parse_bibtex(self, bibtex_str: str) -> List[Dict[str, Any]]:
        """Parse BibTeX string to a list of entries."""
        try:
            bib_database = bibtexparser.loads(bibtex_str)
            return bib_database.entries
        except Exception as e:
            print(f"Error parsing BibTeX: {e}")
            return []
    
    def _format_bibtex(self, entries: List[Dict[str, Any]]) -> str:
        """Format list of entries as BibTeX string."""
        try:
            db = BibDatabase()
            db.entries = entries
            writer = BibTexWriter()
            return writer.write(db)
        except Exception as e:
            print(f"Error formatting BibTeX: {e}")
            return ""
    
    def _detect_reference_format(self, reference: str) -> str:
        """Detect the format of a reference string."""
        reference = reference.strip()
        
        # Check for BibTeX
        if reference.startswith("@") and ("{" in reference or "(" in reference):
            return "bibtex"
        
        # Check for DOI
        doi_pattern = r"10\.\d{4,}/[\w\.-]+"
        if re.search(doi_pattern, reference):
            return "doi"
        
        # Check for IEEE format
        if re.match(r"^\[\d+\]", reference):
            return "ieee"
        
        # Check for APA format (Author, year) pattern
        if re.search(r"\(\d{4}\)", reference):
            return "apa"
        
        # Check for MLA format (Author page) pattern
        if re.search(r"\(\d+\)$", reference):
            return "mla"
        
        # Default to unknown
        return "unknown"
    
    def _validate_bibtex_entry(self, entry: Dict[str, Any]) -> List[str]:
        """Validate a BibTeX entry for common issues."""
        issues = []
        
        # Check required fields based on entry type
        entry_type = entry.get("ENTRYTYPE", "").lower()
        
        required_fields = {
            "article": ["author", "title", "journal", "year"],
            "book": ["author", "title", "publisher", "year"],
            "inproceedings": ["author", "title", "booktitle", "year"],
            "conference": ["author", "title", "booktitle", "year"],
            "phdthesis": ["author", "title", "school", "year"],
            "mastersthesis": ["author", "title", "school", "year"],
            "techreport": ["author", "title", "institution", "year"],
            "misc": ["author", "title", "year"]
        }
        
        # Get required fields for this entry type
        type_required_fields = required_fields.get(entry_type, ["author", "title", "year"])
        
        # Check for required fields
        for field in type_required_fields:
            if field not in entry or not entry[field]:
                issues.append(f"Missing required field: {field}")
        
        # Check for common formatting issues
        if "author" in entry:
            if " and " not in entry["author"] and "," not in entry["author"] and len(entry["author"].split()) > 2:
                issues.append("Author field may have incorrect format, should use 'and' between authors")
        
        if "year" in entry:
            if not re.match(r"^\d{4}$", entry["year"]):
                issues.append("Year should be a 4-digit number")
        
        if "pages" in entry:
            if not re.match(r"^\d+--\d+$", entry["pages"]) and not re.match(r"^\d+$", entry["pages"]):
                issues.append("Pages should be in format '123--456' or single page '123'")
        
        return issues
    
    def _format_citation(self, entry: Dict[str, Any], format_type: str) -> str:
        """Format a citation in the specified format."""
        # Extract common fields
        authors = entry.get("author", "").split(" and ")
        authors = [author.strip() for author in authors]
        
        title = entry.get("title", "").strip()
        # Remove surrounding braces if present
        title = re.sub(r"^\{(.*)\}$", r"\1", title)
        
        year = entry.get("year", "").strip()
        journal = entry.get("journal", "").strip()
        booktitle = entry.get("booktitle", "").strip()
        volume = entry.get("volume", "").strip()
        number = entry.get("number", "").strip()
        pages = entry.get("pages", "").strip()
        publisher = entry.get("publisher", "").strip()
        doi = entry.get("doi", "").strip()
        
        # Format based on specified style
        if format_type == "bibtex":
            db = BibDatabase()
            db.entries = [entry]
            writer = BibTexWriter()
            return writer.write(db).strip()
        
        elif format_type == "apa":
            # Format authors for APA (Last name, First initial.)
            apa_authors = []
            for author in authors:
                if "," in author:
                    # Already in Last, First format
                    parts = author.split(",", 1)
                    last = parts[0].strip()
                    first = parts[1].strip() if len(parts) > 1 else ""
                    # Get initials
                    first_initials = "".join([name[0] + "." for name in first.split()])
                    apa_authors.append(f"{last}, {first_initials}")
                else:
                    # First Last format
                    parts = author.split()
                    if len(parts) >= 2:
                        last = parts[-1]
                        first_initials = "".join([name[0] + "." for name in parts[:-1]])
                        apa_authors.append(f"{last}, {first_initials}")
                    else:
                        apa_authors.append(author)
            
            # Join authors
            if len(apa_authors) == 1:
                author_text = apa_authors[0]
            elif len(apa_authors) == 2:
                author_text = f"{apa_authors[0]} & {apa_authors[1]}"
            elif len(apa_authors) > 2:
                author_text = ", ".join(apa_authors[:-1]) + ", & " + apa_authors[-1]
            else:
                author_text = ""
            
            # Create citation based on entry type
            entry_type = entry.get("ENTRYTYPE", "").lower()
            
            if entry_type == "article":
                return f"{author_text} ({year}). {title}. {journal}, {volume}{f'({number})' if number else ''}, {pages}. {f'https://doi.org/{doi}' if doi else ''}"
            
            elif entry_type in ["inproceedings", "conference"]:
                return f"{author_text} ({year}). {title}. In {booktitle} (pp. {pages}). {publisher}."
            
            elif entry_type == "book":
                return f"{author_text} ({year}). {title}. {publisher}."
            
            else:
                return f"{author_text} ({year}). {title}."
        
        elif format_type == "mla":
            # Format authors for MLA
            if len(authors) == 1:
                author_parts = authors[0].split(",", 1)
                if len(author_parts) > 1:
                    # Last, First format
                    author_text = f"{author_parts[0]}, {author_parts[1].strip()}"
                else:
                    # First Last format
                    parts = authors[0].split()
                    if len(parts) >= 2:
                        author_text = f"{parts[-1]}, {' '.join(parts[:-1])}"
                    else:
                        author_text = authors[0]
            elif len(authors) == 2:
                author_parts1 = authors[0].split(",", 1)
                if len(author_parts1) > 1:
                    # Last, First format
                    first_author = f"{author_parts1[0]}, {author_parts1[1].strip()}"
                else:
                    # First Last format
                    parts = authors[0].split()
                    if len(parts) >= 2:
                        first_author = f"{parts[-1]}, {' '.join(parts[:-1])}"
                    else:
                        first_author = authors[0]
                        
                author_parts2 = authors[1].split(",", 1)
                if len(author_parts2) > 1:
                    # Last, First format for second author
                    second_author = f"{author_parts2[1].strip()} {author_parts2[0]}"
                else:
                    second_author = authors[1]
                    
                author_text = f"{first_author} and {second_author}"
            elif len(authors) > 2:
                author_parts = authors[0].split(",", 1)
                if len(author_parts) > 1:
                    # Last, First format
                    author_text = f"{author_parts[0]}, {author_parts[1].strip()}, et al."
                else:
                    # First Last format
                    parts = authors[0].split()
                    if len(parts) >= 2:
                        author_text = f"{parts[-1]}, {' '.join(parts[:-1])}, et al."
                    else:
                        author_text = f"{authors[0]}, et al."
            else:
                author_text = ""
            
            # Create citation based on entry type
            entry_type = entry.get("ENTRYTYPE", "").lower()
            
            if entry_type == "article":
                return f"{author_text}. \"{title}.\" {journal}, vol. {volume}, no. {number}, {year}, pp. {pages}."
            
            elif entry_type in ["inproceedings", "conference"]:
                return f"{author_text}. \"{title}.\" {booktitle}, {year}, pp. {pages}."
            
            elif entry_type == "book":
                return f"{author_text}. {title}. {publisher}, {year}."
            
            else:
                return f"{author_text}. \"{title}.\" {year}."
        
        elif format_type == "chicago":
            # Format authors for Chicago style
            chicago_authors = []
            for author in authors:
                if "," in author:
                    # Already in Last, First format
                    chicago_authors.append(author)
                else:
                    # First Last format
                    parts = author.split()
                    if len(parts) >= 2:
                        last = parts[-1]
                        first = " ".join(parts[:-1])
                        chicago_authors.append(f"{last}, {first}")
                    else:
                        chicago_authors.append(author)
            
            # Join authors
            if len(chicago_authors) == 1:
                author_text = chicago_authors[0]
            elif len(chicago_authors) == 2:
                author_text = f"{chicago_authors[0]} and {chicago_authors[1]}"
            elif len(chicago_authors) == 3:
                author_text = f"{chicago_authors[0]}, {chicago_authors[1]}, and {chicago_authors[2]}"
            elif len(chicago_authors) > 3:
                author_text = f"{chicago_authors[0]} et al."
            else:
                author_text = ""
            
            # Create citation based on entry type
            entry_type = entry.get("ENTRYTYPE", "").lower()
            
            if entry_type == "article":
                return f"{author_text}. \"{title}.\" {journal} {volume}, no. {number} ({year}): {pages}."
            
            elif entry_type in ["inproceedings", "conference"]:
                return f"{author_text}. \"{title}.\" In {booktitle}, {pages}. {publisher}, {year}."
            
            elif entry_type == "book":
                return f"{author_text}. {title}. {publisher}, {year}."
            
            else:
                return f"{author_text}. \"{title}.\" {year}."
        
        elif format_type == "ieee":
            # Format authors for IEEE (first initial. last)
            ieee_authors = []
            for author in authors:
                if "," in author:
                    # Last, First format
                    parts = author.split(",", 1)
                    last = parts[0].strip()
                    first = parts[1].strip() if len(parts) > 1 else ""
                    # Get initials
                    first_initials = "".join([name[0] + "." for name in first.split()])
                    ieee_authors.append(f"{first_initials} {last}")
                else:
                    # First Last format
                    parts = author.split()
                    if len(parts) >= 2:
                        last = parts[-1]
                        first_initials = "".join([name[0] + "." for name in parts[:-1]])
                        ieee_authors.append(f"{first_initials} {last}")
                    else:
                        ieee_authors.append(author)
            
            # Join authors with commas
            author_text = ", ".join(ieee_authors)
            
            # Create citation based on entry type
            entry_type = entry.get("ENTRYTYPE", "").lower()
            
            if entry_type == "article":
                return f"{author_text}, \"{title},\" {journal}, vol. {volume}, no. {number}, pp. {pages}, {year}."
            
            elif entry_type in ["inproceedings", "conference"]:
                return f"{author_text}, \"{title},\" in {booktitle}, {year}, pp. {pages}."
            
            elif entry_type == "book":
                return f"{author_text}, {title}. {publisher}, {year}."
            
            else:
                return f"{author_text}, \"{title},\" {year}."
        
        elif format_type == "harvard":
            # Format authors for Harvard style
            harvard_authors = []
            for author in authors:
                if "," in author:
                    # Already in Last, First format
                    parts = author.split(",", 1)
                    last = parts[0].strip()
                    first = parts[1].strip() if len(parts) > 1 else ""
                    # Get initials
                    first_initials = "".join([name[0] + "." for name in first.split()])
                    harvard_authors.append(f"{last}, {first_initials}")
                else:
                    # First Last format
                    parts = author.split()
                    if len(parts) >= 2:
                        last = parts[-1]
                        first_initials = "".join([name[0] + "." for name in parts[:-1]])
                        harvard_authors.append(f"{last}, {first_initials}")
                    else:
                        harvard_authors.append(author)
            
            # Join authors
            if len(harvard_authors) == 1:
                author_text = harvard_authors[0]
            elif len(harvard_authors) == 2:
                author_text = f"{harvard_authors[0]} and {harvard_authors[1]}"
            elif len(harvard_authors) > 2:
                author_text = ", ".join(harvard_authors[:-1]) + " and " + harvard_authors[-1]
            else:
                author_text = ""
            
            # Create citation based on entry type
            entry_type = entry.get("ENTRYTYPE", "").lower()
            
            if entry_type == "article":
                return f"{author_text} ({year}). {title}. {journal}, {volume}({number}), pp. {pages}."
            
            elif entry_type in ["inproceedings", "conference"]:
                return f"{author_text} ({year}). {title}. In: {booktitle}, pp. {pages}."
            
            elif entry_type == "book":
                return f"{author_text} ({year}). {title}. {publisher}."
            
            else:
                return f"{author_text} ({year}). {title}."
        
        else:
            return f"Unsupported format: {format_type}"
    
    async def _lookup_reference(self, identifier: str, is_doi: bool = False, is_title: bool = False) -> Dict[str, Any]:
        """Look up reference details from Semantic Scholar."""
        try:
            if is_doi:
                paper = self.ss_client.get_paper(f"DOI:{identifier}", fields=[
                    "paperId", "title", "authors", "year", "abstract", "venue", "externalIds"
                ])
            elif is_title:
                search_results = self.ss_client.search_paper(identifier, limit=1, fields=[
                    "paperId", "title", "authors", "year", "abstract", "venue", "externalIds"
                ])
                if not search_results or not search_results.get("data"):
                    return {"error": "No papers found matching the title"}
                paper = search_results["data"][0]
            else:
                # Assume it's a Semantic Scholar paper ID
                paper = self.ss_client.get_paper(identifier, fields=[
                    "paperId", "title", "authors", "year", "abstract", "venue", "externalIds"
                ])
            
            if not paper:
                return {"error": "Paper not found"}
            
            # Extract info
            title = paper.get("title", "")
            authors = [author.get("name", "") for author in paper.get("authors", [])]
            year = paper.get("year")
            venue = paper.get("venue", "")
            abstract = paper.get("abstract", "")
            doi = paper.get("externalIds", {}).get("DOI")
            
            # Create BibTeX entry
            bibtex_entry = {
                "ENTRYTYPE": "article",
                "ID": f"{authors[0].split()[-1] if authors else ''}_{year}" if year else "unknown",
                "title": title,
                "author": " and ".join(authors),
                "year": str(year) if year else "",
                "journal": venue,
                "abstract": abstract
            }
            
            if doi:
                bibtex_entry["doi"] = doi
            
            # Format citations in different styles
            citations = {
                "bibtex": self._format_citation(bibtex_entry, "bibtex"),
                "apa": self._format_citation(bibtex_entry, "apa"),
                "mla": self._format_citation(bibtex_entry, "mla"),
                "chicago": self._format_citation(bibtex_entry, "chicago"),
                "ieee": self._format_citation(bibtex_entry, "ieee"),
                "harvard": self._format_citation(bibtex_entry, "harvard")
            }
            
            return {
                "title": title,
                "authors": authors,
                "year": year,
                "venue": venue,
                "doi": doi,
                "citations": citations
            }
        
        except Exception as e:
            print(f"Error looking up reference: {e}")
            return {"error": f"Error looking up reference: {str(e)}"}

    async def invoke(self, 
                     action: str = "format",
                     references: Optional[List[str]] = None,
                     bibtex: Optional[str] = None,
                     doi: Optional[str] = None,
                     title: Optional[str] = None,
                     from_format: str = "bibtex",
                     to_format: str = "apa") -> Dict[str, Any]:
        """
        Manage academic references and citations.
        
        Args:
            action: Action to perform with the bibliography
            references: List of references to process
            bibtex: BibTeX format references as a single string
            doi: DOI to look up
            title: Paper title to look up
            from_format: Source citation format
            to_format: Target citation format
            
        Returns:
            Dict containing processed bibliography
        """
        # Handle lookup actions first
        if action == "lookup":
            if doi:
                return {"lookup_results": await self._lookup_reference(doi, is_doi=True)}
            elif title:
                return {"lookup_results": await self._lookup_reference(title, is_title=True)}
            elif references and len(references) == 1:
                # Try to detect the format
                ref_format = self._detect_reference_format(references[0])
                if ref_format == "doi":
                    doi_match = re.search(r"10\.\d{4,}/[\w\.-]+", references[0])
                    if doi_match:
                        return {"lookup_results": await self._lookup_reference(doi_match.group(0), is_doi=True)}
                else:
                    return {"lookup_results": await self._lookup_reference(references[0], is_title=True)}
            else:
                return {"error": "Lookup action requires a DOI, title, or single reference"}
        
        # Handle references list input
        if references:
            if action == "validate":
                # Validate each reference
                validation_results = []
                for i, ref in enumerate(references):
                    ref_format = self._detect_reference_format(ref)
                    if ref_format == "bibtex":
                        entries = self._parse_bibtex(ref)
                        for entry in entries:
                            issues = self._validate_bibtex_entry(entry)
                            if issues:
                                validation_results.append({
                                    "reference_index": i,
                                    "issues": issues
                                })
                    else:
                        validation_results.append({
                            "reference_index": i,
                            "issues": [f"Cannot validate non-BibTeX references. Detected format: {ref_format}"]
                        })
                
                return {"validation_issues": validation_results}
            
            elif action == "format" or action == "convert":
                formatted_refs = []
                for ref in references:
                    ref_format = self._detect_reference_format(ref)
                    
                    # Only handle BibTeX to other formats for now
                    if ref_format == "bibtex" and from_format == "bibtex":
                        entries = self._parse_bibtex(ref)
                        for entry in entries:
                            formatted_refs.append(self._format_citation(entry, to_format))
                    else:
                        # Just pass through references we can't convert
                        formatted_refs.append(ref)
                
                return {"formatted_references": formatted_refs}
        
        # Handle BibTeX string input
        if bibtex:
            if action == "validate":
                entries = self._parse_bibtex(bibtex)
                validation_results = []
                for i, entry in enumerate(entries):
                    issues = self._validate_bibtex_entry(entry)
                    if issues:
                        validation_results.append({
                            "reference_index": i,
                            "issues": issues
                        })
                
                return {"validation_issues": validation_results}
            
            elif action == "format" or action == "convert":
                entries = self._parse_bibtex(bibtex)
                
                if to_format == "bibtex":
                    # Just return the formatted BibTeX
                    return {"bibtex": self._format_bibtex(entries)}
                else:
                    # Convert to the requested format
                    formatted_refs = [self._format_citation(entry, to_format) for entry in entries]
                    return {"formatted_references": formatted_refs}
        
        # If we get here, we couldn't process the input
        return {"error": "Could not process the input. Please check the parameters."}
