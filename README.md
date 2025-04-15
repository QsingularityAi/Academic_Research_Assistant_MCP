# Academic Research Assistant Using Model Context protocal (MCP)

A tool that helps researchers find, analyze, and organize academic papers across multiple databases using Google's Gemini AI and Model Context Protocol (MCP).

## Features

- Natural language research question processing
- Cross-database paper search and retrieval (Google Scholar, Semantic Scholar, arXiv)
- Citation network analysis and visualization
- Research gap identification
- Paper summarization and key findings extraction
- Bibliography management (BibTeX export/import)

## Architecture

This project uses the Model Context Protocol (MCP) to create tools that interact with academic databases and Gemini 2.5 Pro for natural language processing and analysis.

### Components

1. **MCP Tools**
   - Paper Search Tool (cross-database search)
   - Citation Analysis Tool (network generation)
   - Paper Analysis Tool (summarization, key findings)
   - Bibliography Tool (reference management)
   - Research Gap Tool (identifying unexplored areas)

2. **Client Application**
   - Natural language interface
   - Results visualization
   - Bibliography management UI

3. **External APIs**
   - Google Scholar API (via scholarly)
   - Semantic Scholar API
   - arXiv API
   - CrossRef API

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up API keys in a `.env` file
   ```
   GEMINI_API_KEY=your_gemini_api_key
   SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
   ```
4. Run the application: `python client.py`

## Usage

The assistant accepts natural language queries such as:
- "Find recent papers on transformer models in computer vision"
- "Summarize the key findings from this paper: [URL or DOI]"
- "Generate a citation network for papers citing Smith et al. 2022"
- "Identify research gaps in quantum computing cryptography"
- "Create a bibliography for my references on climate change modeling"

## Development

This project is under active development. Contributions are welcome!
