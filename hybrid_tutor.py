import os
import sqlite3
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
DB_PATH = os.path.join(DATA_DIR, "sa_study_app.db")
CHROMA_DIR = os.path.join(DATA_DIR, "sa_textbook_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

print(f"✅ AI Tutor initialized! (Using DB at: {CHROMA_DIR})\n")

def get_platform_resources():
    """Fetches a list of public books and past papers to inject into the AI prompt."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Get public textbooks and past papers
        cursor.execute("SELECT id, title, subject, grade, type FROM books WHERE is_public = 1 OR is_past_paper = 1 LIMIT 30")
        resources = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return resources
    except Exception as e:
        print(f"⚠️ Error fetching platform resources: {e}")
        return []

def hybrid_tutor(question, target_language, book_id=None, chapter=None):
    """
    Answers student questions. Now platform-aware and capable of auto-loading books.
    """
    print(f"🧠 Retrieving context for: '{question}' (Language: {target_language})...")
    
    # Build filter for ChromaDB
    search_filter = {}
    if book_id: search_filter["book_id"] = str(book_id)
    if chapter: search_filter["chapter"] = chapter
        
    search_kwargs = {"k": 4}
    if search_filter: search_kwargs["filter"] = search_filter
    
    retriever = db.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(question)
    
    # Fetch platform resources for the system prompt
    platform_resources = get_platform_resources()
    resources_list = "\n".join([f"- ID: {r['id']}, Title: {r['title']}, Subject: {r['subject']}" for r in platform_resources])
    
    # Generate answer
    if len(docs) > 0 and len(docs[0].page_content) > 50:
        context = "\n\n".join([doc.page_content for doc in docs])
        book_info = docs[0].metadata.get("book_title", "the uploaded document")
        
        prompt = f"""You are Lingo AI, an expert South African tutor and platform guide.
The student asked: "{question}" in {target_language}.

ANSWER USING THIS TEXTBOOK CONTEXT ("{book_info}"):
{context}

CRITICAL RULES:
1. You MUST write your ENTIRE response in {target_language}.
2. If the user asks what books are on the platform, list them from the context or general knowledge.
3. If the user asks to "load", "open", or "use" a specific book from the platform, you MUST end your response with the exact tag: [LOAD_BOOK:{book_id}]. (Replace {book_id} with the ID of the book they want).
4. Keep the tone friendly and educational.

ANSWER IN {target_language}:"""
        source_type = "textbook"
        
    else:
        # Fallback prompt if no specific book context is found (Platform Guide Mode)
        prompt = f"""You are Lingo AI, an expert South African tutor and platform guide.
The student asked: "{question}" in {target_language}.

AVAILABLE RESOURCES ON PLATFORM:
{resources_list if resources_list else "No public resources currently uploaded."}

HOW TO USE THE PLATFORM:
- AI Tutor: Search for a book in the search bar to attach it to the chat, then ask questions.
- Library: Browse public textbooks.
- Past Papers: View and download past exam papers.
- Upload: Upload your own notes (select 'Past Paper (Public)' to share with everyone).
- Study Tools: Generate quizzes from your documents.

CRITICAL RULES:
1. You MUST write your ENTIRE response in {target_language}.
2. Answer the user's question about the platform or general knowledge.
3. If the user asks to "load", "open", or "use" a specific book listed above, you MUST end your response with the exact tag: [LOAD_BOOK:ID]. (Replace ID with the actual number from the list).
4. Do NOT explain the tag to the user. Just output it at the very end.

ANSWER IN {target_language}:"""
        source_type = "general knowledge"
    
    print(f"✅ Generating response in {target_language}...")
    response = llm.invoke(prompt)
    
    return response.content, source_type