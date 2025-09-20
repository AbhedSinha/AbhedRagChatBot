import sqlite3
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage


DB_NAME = "rag_app.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    with get_db_connection() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         session_id TEXT,
                         user_query TEXT,
                         gpt_response TEXT,
                         model TEXT,
                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

def insert_application_logs(session_id, user_query, gpt_response, model):
    with get_db_connection() as conn:
        conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                     (session_id, user_query, gpt_response, model))

def get_chat_history(session_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
        messages = []
        for row in cursor.fetchall():
            messages.append(HumanMessage(content=row['user_query']))
            messages.append(AIMessage(content=row['gpt_response']))
        return messages

def create_document_store():
    with get_db_connection() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         filename TEXT,
                         upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

def insert_document_record(filename):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO document_store (filename) VALUES (?)', (filename,))
        return cursor.lastrowid

def delete_document_record(file_id):
    with get_db_connection() as conn:
        conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    return True

def get_all_documents():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
        documents = cursor.fetchall()
        return [dict(doc) for doc in documents]

# Initialize the database tables
create_application_logs()
create_document_store()