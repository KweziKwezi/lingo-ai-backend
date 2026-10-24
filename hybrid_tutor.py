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

# 2. Load the textbook vector database
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory="./sa_textbook_db", embedding_function=embeddings)

print("✅ AI Tutor initialized!\n")

def hybrid_tutor(question, target_language, book_id=None, chapter=None):
    """
    Answers student questions using a specific book's context.
    The student can type in their local language, and the AI will reply in that language.
    """
    print(f"🧠 Retrieving context for: '{question}' (Language: {target_language})...")
    
    # Build filter using ChromaDB's required syntax ($eq and $and)
    conditions = []
    if book_id:
        conditions.append({"book_id": {"$eq": str(book_id)}})
    if chapter:
        conditions.append({"chapter": {"$eq": chapter}})
    
    search_filter = {}
    if len(conditions) == 1:
        search_filter = conditions[0]
    elif len(conditions) > 1:
        search_filter = {"$and": conditions}
    
    # Retrieve relevant chunks
    search_kwargs = {"k": 3}
    if search_filter:
        search_kwargs["filter"] = search_filter
    
    retriever = db.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(question)
    
    # Generate answer
    if len(docs) > 0 and len(docs[0].page_content) > 100:
        context = "\n\n".join([doc.page_content for doc in docs])
        book_info = docs[0].metadata.get("book_title", "the textbook")
        
        prompt = f"""You are an expert South African high school tutor.
The student has asked you a question in {target_language}. 
They might have typed the question in English, or they might have typed it directly in {target_language}.

Answer their question using the context from "{book_info}" below.
If the context doesn't fully answer the question, supplement with your general knowledge.

CRITICAL RULE: You MUST write your ENTIRE response in {target_language}.
Keep the tone friendly, clear, and educational.

TEXTBOOK CONTEXT (in English):
{context}

STUDENT QUESTION: {question}

ANSWER IN {target_language}:"""
        source_type = "textbook"
        
    else:
        prompt = f"""You are an expert South African high school tutor.
The student has asked you a question in {target_language}, but the answer is not in their textbook.
Answer using your general knowledge.

CRITICAL RULE: You MUST write your ENTIRE response in {target_language}.
Keep the tone friendly, clear, and educational.

STUDENT QUESTION: {question}

ANSWER IN {target_language}:"""
        source_type = "general knowledge"
    
    print(f"✅ Generating response in {target_language}...")
    response = llm.invoke(prompt)
    
    return response.content, source_type