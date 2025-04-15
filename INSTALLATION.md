# Academic Research Assistant - Installation Guide

This guide will help you set up and run the Academic Research Assistant on your local machine.

## Prerequisites

Before you begin, you'll need:

1. Python 3.9 or newer
2. [Google Gemini API Key](https://aistudio.google.com/app/apikey)
3. [Semantic Scholar API Key](https://www.semanticscholar.org/product/api) (optional but recommended)

## Installation Steps

### 1. Clone or download the repository

Download and extract the ZIP file or clone the repository to your local machine.

### 2. Create a virtual environment (recommended)

```bash
# Navigate to the project directory
cd academic-research-assistant

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Open the `.env` file in a text editor and add your API keys:

```
GEMINI_API_KEY=your_gemini_api_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here
```

### 5. Make server scripts executable (macOS/Linux only)

```bash
chmod +x mcp_server.py
chmod +x client.py
```

## Running the Assistant

To start the Academic Research Assistant:

```bash
python client.py
```

The client will:
1. Start the MCP server in the background
2. Connect to the server via stdio
3. Present an interactive interface for your research queries

## Example Queries

Here are some examples of queries you can try:

1. **Paper Search**:
   - "Find recent papers on transformer models in computer vision"
   - "Search for papers about climate change published in Nature since 2020"

2. **Citation Analysis**:
   - "Analyze the citation network for the paper with DOI 10.1038/s41586-020-2649-2"
   - "Generate a citation graph for papers about COVID-19 vaccines"

3. **Paper Analysis**:
   - "Summarize the key findings from this paper: [DOI or title]"
   - "Extract the methodology from this arXiv paper: [arXiv ID]"

4. **Bibliography Management**:
   - "Convert this BibTeX reference to APA format: [BibTeX citation]"
   - "Look up the citation information for this DOI: [DOI]"

5. **Research Gap Identification**:
   - "Identify research gaps in quantum computing cryptography"
   - "What are the future research directions for federated learning in healthcare?"

## Troubleshooting

If you encounter any issues:

1. **API Key Problems**: Ensure your API keys are correctly set in the `.env` file.

2. **Dependency Issues**: Make sure all dependencies are installed correctly with:
   ```bash
   pip install -r requirements.txt
   ```

3. **Server Connection Issues**: If the client can't connect to the server, try running the server separately:
   ```bash
   python mcp_server.py --connection_type stdio
   ```
   Then in another terminal:
   ```bash
   python client.py
   ```

4. **Rate Limiting**: Both Semantic Scholar and Google Gemini APIs have rate limits. If you encounter errors, wait a few minutes and try again.

## Next Steps

Once you're comfortable with the basic usage, you can:

1. Modify the MCP tools to add more features
2. Add new data sources beyond Semantic Scholar
3. Create a web interface for the assistant
4. Integrate with reference management software
