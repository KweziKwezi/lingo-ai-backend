import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

print("🔄 Initializing SA Study AI Tutor...")

# 1. Initialize OpenAI LLM
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)

# 2. Load the textbook vector database from the persistent disk
DATA_DIR = "/data" if os.path.exists("/data") else "."
CHROMA_DIR = os.path.join(DATA_DIR, "sa_textbook_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

print(f"✅ AI Tutor initialized! (Using DB at: {CHROMA_DIR})\n")

def hybrid_tutor(question, target_language, book_id=None, chapter=None):
    """
    Answers student questions using a specific book's context.
    """
    print(f" Retrieving context for: '{question}' (Language: {target_language})...")
    print(f"🔍 Input book_id: {book_id} (Type: {type(book_id)})")
    
    # Build filter using simple key-value pairs (Removes $eq to prevent silent ChromaDB failures)
    search_filter = {}
    if book_id:
        search_filter["book_id"] = str(book_id)
    if chapter:
        search_filter["chapter"] = chapter
        
    print(f" Applying ChromaDB filter: {search_filter}")
    
    # Retrieve relevant chunks
    search_kwargs = {"k": 4} # Increased from 3 to get a bit more context
    if search_filter:
        search_kwargs["filter"] = search_filter
    
    retriever = db.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(question)
    
    print(f"🔍 Found {len(docs)} documents in ChromaDB.")
    
    # Generate answer
    if len(docs) > 0 and len(docs[0].page_content) > 50:
        context = "\n\n".join([doc.page_content for doc in docs])
        book_info = docs[0].metadata.get("book_title", "the uploaded document")
        
        prompt = f"""You are an expert South African tutor.
The student has asked you a question in {target_language}. 

Answer their question using ONLY the context from "{book_info}" below.
If the context doesn't fully answer the question, use your general knowledge but mention that the specific detail wasn't in the text.

CRITICAL RULE: You MUST write your ENTIRE response in {target_language}.
Keep the tone friendly, clear, and educational.

TEXTBOOK CONTEXT:
{context}

STUDENT QUESTION: {question}

ANSWER IN {target_language}:"""
        source_type = "textbook"
        
    else:
        # Fallback prompt if no documents are found
        prompt = f"""You are an expert South African tutor.
The student has asked you a question in {target_language}. 
I could not find specific information about this in their uploaded document.

Answer using your general knowledge, but start your response by politely saying: 
"I couldn't find that specific detail in your uploaded notes, but generally speaking..."

CRITICAL RULE: You MUST write your ENTIRE response in {target_language}.
Keep the tone friendly, clear, and educational.

STUDENT QUESTION: {question}

ANSWER IN {target_language}:"""
        source_type = "general knowledge"
    
    print(f"✅ Generating response in {target_language}...")
    response = llm.invoke(prompt)
    
    return response.content, source_type