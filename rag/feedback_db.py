import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/feedback.db")

def init_db():
    """Initialize the SQLite database for storing report feedback."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS report_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            transcript TEXT,
            original_report TEXT NOT NULL,
            edited_report TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def save_feedback(transcript: str, original_report: str, edited_report: str):
    """Save a user-corrected report to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO report_feedback (timestamp, transcript, original_report, edited_report)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, transcript, original_report, edited_report))
    
    conn.commit()
    conn.close()

def get_all_feedback():
    """Retrieve all feedback records."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, timestamp, transcript, original_report, edited_report FROM report_feedback ORDER BY id DESC')
    rows = cursor.fetchall()
    
    conn.close()
    
    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "transcript": r[2],
            "original_report": r[3],
            "edited_report": r[4]
        }
        for r in rows
    ]

# Initialize DB on import
init_db()
