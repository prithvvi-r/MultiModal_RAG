from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import json

load_dotenv()

# CRITICAL FIX: Load the existing vector store
def load_vector_store(persist_directory="AdvRag/db/chroma_db"):
    """Load existing ChromaDB vector store"""
    print(f" Loading vector store from {persist_directory}")
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f" Vector store loaded with {vectorstore._collection.count()} documents")
    return vectorstore


def generate_final_answer(chunks, query):
    """Generate final answer using multimodal content"""
    
    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question: {query}

RETRIEVED DOCUMENTS:
"""
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"\n--- Document {i+1} ---\n"
            
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                
                # Add raw text
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                
                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            else:
                # Fallback to page_content if no original_content
                prompt_text += f"CONTENT:\n{chunk.page_content}\n\n"
            
        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. 
If the documents don't contain sufficient information, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add all images from all chunks
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
                
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f" Answer generation failed: {e}")
        return f"Sorry, I encountered an error while generating the answer: {str(e)}"


def retrieve_and_answer(query, persist_directory="AdvRag/db/chroma_db", k=3):
    """Complete RAG retrieval pipeline"""
    print(f" Query: {query}\n")
    
    # Step 1: Load vector store
    db = load_vector_store(persist_directory)
    
    # Step 2: Create retriever and get relevant chunks
    print(f" Retrieving top {k} relevant chunks...")
    retriever = db.as_retriever(search_kwargs={"k": k})
    chunks = retriever.invoke(query)
    print(f" Retrieved {len(chunks)} chunks\n")
    
    # Step 3: Generate answer
    print(" Generating answer...")
    final_answer = generate_final_answer(chunks, query)
    
    return final_answer, chunks


# Main execution
def main():
    query = "According to table 1, what are the main advantages of self attention layesrs comapare to reccurent and convulational"
    
    answer, chunks = retrieve_and_answer(
        query=query,
        persist_directory="AdvRag/db/chroma_db",
        k=3
    )
    
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(answer)
    print("="*50)


if __name__ == '__main__':
    main()