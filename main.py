   # FORCE GIT UPDATE
import os
import shutil
import json
import re
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

# LangChain imports for Quiz generation
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Local imports
from hybrid_tutor import hybrid_tutor
from ingest_textbook import ingest_textbook
from database import (
    get_all_books, get_book_by_id, delete_book,
    create_user, get_user_by_email, get_user_by_id
)
from auth import (
    verify_password, get_password_hash, create_access_token,
    get_current_user
)

load_dotenv()

# Create uploads folder if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

app = FastAPI(
    title="SA Study App API",
    description="AI tutor for South African students with authentication",
    version="3.0.0"
)

# ==========================================
# CORS MIDDLEWARE
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# REQUEST/RESPONSE MODELS
# ==========================================
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    id: int
    email: str
    name: str

class QuestionRequest(BaseModel):
    question: str
    language: str
    book_id: Optional[int] = None
    chapter: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    source: str

class BookResponse(BaseModel):
    id: int
    title: str
    grade: str
    subject: str
    filename: str
    total_chunks: int
    type: str
    uploaded_at: str

class QuizGenerateRequest(BaseModel):
    book_id: int
    num_questions: int = 5

class QuizGradeRequest(BaseModel):
    questions: list
    user_answers: dict  # MUST be dict to match frontend

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def extract_json_from_llm(text: str):
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1:
        return json.loads(text[start:end+1])
    return json.loads(text)

# ==========================================
# BASIC ENDPOINTS
# ==========================================
@app.get("/")
def home():
    return {
        "message": "🇿🇦 SA Study App API v3.0",
        "features": [
            "User authentication",
            "Multi-textbook library",
            "Native language typing & responding",
            "PDF and PPTX upload"
        ]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# ==========================================
# AUTHENTICATION ENDPOINTS
# ==========================================
@app.post("/register", response_model=UserResponse)
def register(user: UserRegister):
    existing_user = get_user_by_email(user.email)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    
    password_hash = get_password_hash(user.password)
    user_id = create_user(email=user.email, password_hash=password_hash, name=user.name)
    
    if not user_id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create user")
    
    return {"id": user_id, "email": user.email, "name": user.name}

@app.post("/login", response_model=Token)
def login(user: UserLogin):
    db_user = get_user_by_email(user.email)
    if not db_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    
    if not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": str(db_user["id"])})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me", response_model=UserResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    return {"id": current_user["id"], "email": current_user["email"], "name": current_user["name"]}

# ==========================================
# LIBRARY ENDPOINTS (Authenticated)
# ==========================================
@app.get("/books", response_model=List[BookResponse])
def list_books(my_books_only: bool = False, current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"] if my_books_only else None
    books = get_all_books(user_id=user_id)
    return books

@app.get("/books/{book_id}", response_model=BookResponse)
def get_book(book_id: int, current_user: dict = Depends(get_current_user)):
    book = get_book_by_id(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book

@app.delete("/books/{book_id}")
def remove_book(book_id: int, current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    success = delete_book(book_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Book not found or not owned by you")
    return {"message": "Book removed from library"}

# ==========================================
# UPLOAD ENDPOINT (Authenticated)
# ==========================================
@app.post("/upload")
async def upload_textbook(
    file: UploadFile = File(...),
    title: str = Form(...),
    grade: str = Form(...),
    subject: str = Form(...),
    type: str = Form(default="textbook"),
    current_user: dict = Depends(get_current_user)
):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".pptx")):
        raise HTTPException(status_code=400, detail="Only PDF and PPTX files are allowed")

    if type not in ["textbook", "slides", "notes"]:
        raise HTTPException(status_code=400, detail="Type must be 'textbook', 'slides', or 'notes'")

    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        print(f"\n🚀 Starting ingestion for: {title}...")
        book_id = ingest_textbook(
            file_path=file_path,
            title=title,
            grade=grade,
            subject=subject,
            user_id=current_user["id"],
            type=type
        )
        return {
            "message": "File successfully uploaded and added to the library!", 
            "book_id": book_id, 
            "title": title, 
            "type": type
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

# ==========================================
# ASK THE TUTOR ENDPOINT (Authenticated)
# ==========================================
@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest, current_user: dict = Depends(get_current_user)):
    try:
        print(f"\n📩 New question from user {current_user['email']}:")
        print(f"   Question: {request.question}")
        print(f"   Language: {request.language}")
        print(f"   Book ID: {request.book_id}")
        
        answer, source = hybrid_tutor(
            question=request.question,
            target_language=request.language,
            book_id=request.book_id,
            chapter=request.chapter
        )
        return QuestionResponse(answer=answer, source=source)
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# QUIZ & FLASHCARD ENDPOINTS (Authenticated)
# ==========================================
@app.post("/generate-quiz")
def generate_quiz(req: QuizGenerateRequest, current_user: dict = Depends(get_current_user)):
    try:
        print(f"🔍 [QUIZ] Attempting to generate quiz for book_id: {req.book_id}")
        
        book = get_book_by_id(req.book_id)
        if not book:
            print(f"❌ [QUIZ] Book not found for id: {req.book_id}")
            raise HTTPException(status_code=404, detail="Book not found")
        
        if isinstance(book, dict):
            b_id = book.get("id")
        else:
            b_id = getattr(book, "id", None)
            
        if not b_id:
            print(f"❌ [QUIZ] Could not determine book ID from: {book}")
            raise HTTPException(status_code=500, detail="Could not determine book ID")
            
        print(f"✅ [QUIZ] Found book ID: {b_id}. Loading Chroma DB...")

        db_path = "./sa_textbook_db"
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        retriever = vectorstore.as_retriever(search_kwargs={
            "k": 8,
            "filter": {"book_id": str(b_id)}
        })
        
        docs = retriever.invoke("main concepts and details")
        print(f"✅ [QUIZ] Retrieved {len(docs)} chunks of context for book {b_id}.")
        
        if len(docs) == 0:
            raise HTTPException(status_code=400, detail="No content found for this book. Please ensure it was uploaded successfully.")

        context = "\n\n".join([doc.page_content for doc in docs])

        print(f"🤖 [QUIZ] Calling OpenAI to generate {req.num_questions} UNIQUE questions...")
        llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")

        prompt = f"""You are an expert teacher. Based ONLY on the context below, generate {req.num_questions} questions.
        
        CRITICAL INSTRUCTION: These questions MUST BE UNIQUE and varied. Do not repeat the same questions. Focus on different facts, definitions, and concepts from the text each time.
        The questions MUST be a mix of these three types: "mcq", "short", and "long".

        Context: {context}

        Output ONLY a valid JSON array of objects. Each object must have:
        - "id": integer (1 to {req.num_questions})
        - "type": string ("mcq", "short", or "long")
        - "question": string
        - "options": array of 4 strings (ONLY for "mcq", else null)
        - "correct_answer": integer 0-3 (ONLY for "mcq", else null)
        - "model_answer": string (The correct answer/explanation for grading short/long)
        - "marks": integer (1 for mcq/short, 3 for long)

        Do NOT wrap the output in ```json ... ``` markdown blocks. Just output the raw JSON array."""

        response = llm.invoke(prompt)
        print("✅ [QUIZ] OpenAI responded. Parsing JSON...")
        
        parsed_data = extract_json_from_llm(response.content)
        print("✅ [QUIZ] Quiz generated successfully!")
        return {"questions": parsed_data}
        
    except json.JSONDecodeError as e:
        print(f"❌ [QUIZ] JSON Decode Error: {e}")
        raise HTTPException(status_code=500, detail="AI failed to generate valid JSON format.")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [QUIZ] CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/grade-quiz")
def grade_quiz(req: QuizGradeRequest, current_user: dict = Depends(get_current_user)):
    try:
        llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")

        prompt = f"""You are an expert teacher grading a quiz. 
        Here are the original questions and the student's answers.
        
        Questions: {json.dumps(req.questions)}
        Student Answers: {json.dumps(req.user_answers)}

        Grade the student's work. For MCQs, check if the index matches 'correct_answer'. 
        For short/long answers, compare their answer to the 'model_answer' and award partial marks if they are close. If the student left an answer blank, award 0 marks and note it.

        Output ONLY a valid JSON object with this EXACT structure:
        {{
          "total_score": integer,
          "max_score": integer,
          "graded_questions": [
            {{
              "id": integer,
              "is_correct": boolean,
              "awarded_marks": integer,
              "feedback": "string explaining why it is right or wrong, be encouraging"
            }}
          ]
        }}
        Do NOT wrap the output in ```json ... ``` markdown blocks."""

        response = llm.invoke(prompt)
        parsed_data = extract_json_from_llm(response.content)
        
        if not isinstance(parsed_data, dict) or "graded_questions" not in parsed_data:
            raise ValueError("AI returned invalid grading structure")
            
        return parsed_data
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI failed to generate valid JSON format for grading.")
    except Exception as e:
        print(f"❌ [GRADE] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")

# ==========================================
# RUN THE SERVER
# ==========================================
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting SA Study App API server...")
    print("📖 API Documentation: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)