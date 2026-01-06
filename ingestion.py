import os
import sys
import json
from typing import List

# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# LangChain components
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Add this BEFORE your partition_pdf call
os.environ['TESSDATA_PREFIX'] = r'C:\Users\PRUTHVIRAJ\AppData\Local\Programs\Tesseract-OCR\tessdata'


def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    print(f" Partitioning document: {file_path}")
    
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )
    
    print(f" Extracted {len(elements)} elements")
    return elements


def create_chunks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    print(" Creating smart chunks...")
    
    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )
    
    print(f" Created {len(chunks)} chunks")
    return chunks


def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            # Handle images
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data


def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""
    
    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Build the text prompt
        prompt_text = f"""You are creating a searchable description for document content retrieval.

CONTENT TO ANALYZE:
TEXT CONTENT:
{text}

"""
        
        # Add tables if present
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
        prompt_text += """
YOUR TASK:
Generate a comprehensive, searchable description that covers:

1. Key facts, numbers, and data points from text and tables
2. Main topics and concepts discussed  
3. Questions this content could answer
4. Visual content analysis (charts, diagrams, patterns in images)
5. Alternative search terms users might use

Make it detailed and searchable - prioritize findability over brevity.

SEARCHABLE DESCRIPTION:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add images to the message
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f"      AI summary failed: {e}")
        # Fallback to simple summary
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


def summarise_chunks(chunks):
    """Process all chunks with AI Summaries"""
    print(" Processing chunks with AI Summaries...")
    
    langchain_documents = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"   Processing chunk {current_chunk}/{total_chunks}")
        
        # Analyze chunk content
        content_data = separate_content_types(chunk)
        
        # Debug prints
        print(f"     Types found: {content_data['types']}")
        print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")
        
        # Create AI-enhanced summary if chunk has tables/images
        if content_data['tables'] or content_data['images']:
            print(f"     → Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'], 
                    content_data['images']
                )
                print(f"     → AI summary created successfully")
                print(f"     → Enhanced content preview: {enhanced_content[:200]}...")
            except Exception as e:
                print(f"      AI summary failed: {e}")
                enhanced_content = content_data['text']
        else:
            print(f"     → Using raw text (no tables/images)")
            enhanced_content = content_data['text']
        
        # Create LangChain Document with rich metadata
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                })
            }
        )
        
        langchain_documents.append(doc)
    
    print(f" Processed {len(langchain_documents)} chunks")
    return langchain_documents


def export_chunks_to_json(chunks, filename="chunks_export.json"):
    """Export processed chunks to clean JSON format"""
    export_data = []
    
    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}"))
            }
        }
        export_data.append(chunk_data)
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f" Exported {len(export_data)} chunks to {filename}")
    return export_data


def create_vector_store(documents, persist_directory="AdvRag/db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print(" Creating embeddings and storing in ChromaDB...")
        
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f" Vector store created and saved to {persist_directory}")
    return vectorstore


# ONLY run this if the file is executed directly (not imported)
if __name__ == '__main__':
    # Test execution - only runs when you do: python ingestion.py
    file_path = "./docs/attention-is-all-you-need.pdf"
    elements = partition_document(file_path)
    chunks = create_chunks_by_title(elements)
    processed_chunks = summarise_chunks(chunks)
    json_data = export_chunks_to_json(processed_chunks)
    db = create_vector_store(processed_chunks)