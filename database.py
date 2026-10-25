import sqlite3
import os

DB_PATH = "sa_study_app.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            grade TEXT NOT NULL,
            subject TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            total_chunks INTEGER NOT NULL,
            type TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            book_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (book_id) REFERENCES books(id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            questions TEXT NOT NULL,
            user_answers TEXT NOT NULL,
            results TEXT NOT NULL,
            score INTEGER NOT NULL,
            max_score INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    
    conn.commit()
    conn.close()

def create_user(email: str, password_hash: str, name: str) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)", (email, password_hash, name))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_user_by_email(email: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def get_user_by_id(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def add_book(title: str, grade: str, subject: str, filename: str, file_path: str, total_chunks: int, user_id: int, type: str) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO books (user_id, title, grade, subject, filename, file_path, total_chunks, type) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (user_id, title, grade, subject, filename, file_path, total_chunks, type))
    book_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return book_id

def get_all_books(user_id: int = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_id:
        cursor.execute("SELECT * FROM books WHERE user_id = ? ORDER BY uploaded_at DESC", (user_id,))
    else:
        cursor.execute("SELECT * FROM books ORDER BY uploaded_at DESC")
    books = cursor.fetchall()
    conn.close()
    return [dict(book) for book in books]

def get_book_by_id(book_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM books WHERE id = ?", (book_id,))
    book = cursor.fetchone()
    conn.close()
    return dict(book) if book else None

def delete_book(book_id: int, user_id: int) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM books WHERE id = ? AND user_id = ?", (book_id, user_id))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def create_session(user_id: int, book_id: int, title: str) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sessions (user_id, book_id, title) VALUES (?, ?, ?)", (user_id, book_id, title))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def get_user_sessions(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, b.title as book_title, b.subject, b.grade
        FROM sessions s
        JOIN books b ON s.book_id = b.id
        WHERE s.user_id = ?
        ORDER BY s.created_at DESC
    """, (user_id,))
    sessions = cursor.fetchall()
    conn.close()
    return [dict(row) for row in sessions]

def save_quiz_attempt(session_id: int, questions: str, answers: str, results: str, score: int, max_score: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO quiz_attempts (session_id, questions, user_answers, results, score, max_score) VALUES (?, ?, ?, ?, ?, ?)", (session_id, questions, answers, results, score, max_score))
    conn.commit()
    conn.close()

def get_session_quiz_history(session_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM quiz_attempts WHERE session_id = ? ORDER BY created_at DESC", (session_id,))
    attempts = cursor.fetchall()
    conn.close()
    return [dict(row) for row in attempts]

init_db()
