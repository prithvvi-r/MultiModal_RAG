# Multimodal RAG System 🚀

A powerful Retrieval-Augmented Generation (RAG) system that handles text, tables, and images from PDF documents using AI-enhanced chunking and multimodal LLM capabilities.

## Overview

This system intelligently processes PDF documents by:
- Extracting text, tables, and images using Unstructured
- Creating smart chunks with title-based segmentation
- Generating AI-enhanced summaries for multimodal content
- Storing embeddings in ChromaDB for efficient retrieval
- Answering queries using GPT-4o with full multimodal context

## Architecture

```
PDF Document
    ↓
[Unstructured Parser] → Extract text, tables, images
    ↓
[Title-Based Chunking] → Create intelligent segments
    ↓
[AI Enhancement] → Generate searchable summaries (GPT-4o)
    ↓
[ChromaDB] → Store embeddings + original content
    ↓
[Query] → Retrieve relevant chunks
    ↓
[GPT-4o] → Generate answer with full context
```

## Features

- **Multimodal Processing**: Handles text, tables (HTML), and images
- **Intelligent Chunking**: Title-based segmentation preserves context
- **AI-Enhanced Summaries**: GPT-4o creates searchable descriptions
- **Dual Storage**: Embeddings for search + original content for generation
- **Vision-Enabled Answers**: GPT-4o analyzes images during retrieval
- **Structured Export**: JSON export of all processed chunks

## Installation

### Prerequisites

- Python 3.8+
- Poppler installed
- Tesseract OCR installed
- OpenAI API key

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/prithvvi-r/MultiModal_RAG.git
cd multimodal-rag
```
2.**Install Poppler**
   - `https://github.com/oschwartz10612/poppler-windows`
   - for mac and linux visit `https://pypi.org/project/python-poppler/`

2. **Install Tesseract OCR**
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Mac: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

3. **Install Python dependencies**
```bash
pip install unstructured
pip install "unstructured[all_docs]"
pip install langchain-openai langchain-chroma langchain-core
pip install python-dotenv
pip install chromadb
```

4. **Set up environment variables**

Create a `.env` file:
```env
OPENAI_API_KEY=your_api_key_here
```

5. **Configure Tesseract path** (Windows only)

Update line 18 in `ingestion.py`:
```python
os.environ['TESSDATA_PREFIX'] = r'C:\path\to\your\Tesseract-OCR\tessdata'
```

## Usage

### 1. Document Ingestion

Process a PDF and create the vector database:

```python
python ingestion.py
```

**What happens:**
- Extracts all content from PDF
- Creates ~3000 character chunks
- Generates AI summaries for chunks with tables/images
- Stores embeddings in `db/chroma_db`
- Exports chunks to `chunks_export.json`

**Customize the source document:**
```python
# In ingestion.py, line 203
file_path = "./docs/your-document.pdf"
```

### 2. Query and Retrieval

Ask questions about your documents:

```python
python retrieval.py
```

**Customize your query:**
```python
# In retrieval.py, line 109
query = "What are the key findings in Table 1?"
```

**Adjust retrieval parameters:**
```python
answer, chunks = retrieve_and_answer(
    query=query,
    persist_directory="AdvRag/db/chroma_db",
    k=3  # Number of chunks to retrieve
)
```

## Project Structure

```
multimodal-rag/
├── ingestion.py          # Document processing pipeline
├── retrieval.py          # Query and answer generation
├── docs/                 # PDF documents to process
├── db/
│       └── chroma_db/    # Vector database storage
├── chunks_export.json    # Exported processed chunks
├── .env                  # API keys (create this)
└── README.md
```

## Key Components

### Ingestion Pipeline (`ingestion.py`)

| Function | Purpose |
|----------|---------|
| `partition_document()` | Extracts elements from PDF with hi-res strategy |
| `create_chunks_by_title()` | Creates 3000-char chunks with title awareness |
| `separate_content_types()` | Identifies text, tables, and images in chunks |
| `create_ai_enhanced_summary()` | Generates GPT-4o summaries for multimodal chunks |
| `summarise_chunks()` | Processes all chunks with AI enhancement |
| `create_vector_store()` | Creates ChromaDB with embeddings |
| `export_chunks_to_json()` | Exports processed data to JSON |

### Retrieval Pipeline (`retrieval.py`)

| Function | Purpose |
|----------|---------|
| `load_vector_store()` | Loads existing ChromaDB instance |
| `generate_final_answer()` | Uses GPT-4o with text, tables, and images |
| `retrieve_and_answer()` | Complete RAG pipeline |

## Configuration

### Chunking Parameters
```python
chunk_by_title(
    elements,
    max_characters=3000,      # Maximum chunk size
    new_after_n_chars=2400,   # Soft limit before breaking
    combine_text_under_n_chars=500  # Combine small chunks
)
```

### Models Used
- **Embeddings**: `text-embedding-3-small` (OpenAI)
- **Summarization**: `gpt-4o` (multimodal)
- **Answer Generation**: `gpt-4o-mini` (multimodal)

### Vector Store Settings
- **Distance Metric**: Cosine similarity
- **Persistence**: Local ChromaDB storage

## Example Workflow

```python
# 1. Process a document
from ingestion import *

file_path = "./docs/research-paper.pdf"
elements = partition_document(file_path)
chunks = create_chunks_by_title(elements)
processed_chunks = summarise_chunks(chunks)
db = create_vector_store(processed_chunks)

# 2. Query the document
from retrieval import retrieve_and_answer

query = "What methodology did the authors use?"
answer, chunks = retrieve_and_answer(query, k=5)
print(answer)
```

## Advanced Features

### JSON Export Format
```json
{
  "chunk_id": 1,
  "enhanced_content": "AI-generated searchable summary...",
  "metadata": {
    "original_content": {
      "raw_text": "Original text...",
      "tables_html": ["<table>...</table>"],
      "images_base64": ["base64_encoded_image..."]
    }
  }
}
```

### Error Handling
- Graceful fallback if AI summarization fails
- Preserves original content even when enhancement unavailable
- Informative progress tracking during processing

## Limitations

- Requires OpenAI API access (GPT-4o for best results)
- Processing time scales with document size and number of images
- Tesseract OCR quality depends on image clarity
- Token limits apply to very large documents

## Troubleshooting

**Issue**: `TESSDATA_PREFIX` error
- **Solution**: Install Tesseract OCR and set correct path in `ingestion.py`

**Issue**: ChromaDB not found
- **Solution**: Run `ingestion.py` first to create the database

**Issue**: Out of tokens error
- **Solution**: Reduce `k` parameter or chunk size

**Issue**: Slow processing
- **Solution**: Reduce image quality in partition_pdf or limit `extract_image_block_types`

## Future Enhancements

- [ ] Support for multiple document formats (DOCX, HTML)
- [ ] Batch processing for multiple PDFs
- [ ] Custom embedding models
- [ ] Web interface for queries
- [ ] Semantic chunking strategies
- [ ] Query result caching

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## License

MIT License - feel free to use and modify

## Acknowledgments

- Built with [Unstructured](https://unstructured.io/) for document parsing
- Powered by [LangChain](https://langchain.com/) and [ChromaDB](https://www.trychroma.com/)
- Uses OpenAI GPT-4o for multimodal understanding

---

**Questions?** Open an issue or reach out to the maintainers.
