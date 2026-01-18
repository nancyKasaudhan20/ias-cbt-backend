
# -----------------------NANCY------------------------------
import os
import re
import json
import uuid
import string
import pdfplumber
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Load local .env file (for development only)
load_dotenv()
# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-Memory DB ---
TEMP_DB = {}

# --- Models ---
class Option(BaseModel):
    id: str
    text: str

class QuestionPublic(BaseModel):
    id: int
    text: str
    options: List[Option]

class TestResponse(BaseModel):
    test_id: str
    questions: List[QuestionPublic]

class Submission(BaseModel):
    test_id: str
    user_answers: Dict[str, str]

class QuestionResult(BaseModel):
    question_id: int
    question_text: str
    options: List[Option]
    user_selected: Optional[str]
    correct_option: str
    is_correct: bool
    explanation: str

class ResultResponse(BaseModel):
    score: int
    total: int
    percentage: float
    details: List[QuestionResult]

# ==========================================
# HELPER: Pre-Cleaning (Crucial for Speed)
# ==========================================
def clean_text_common(text: str) -> str:
    """
    Removes Hindi and non-printable chars immediately.
    This saves token space for the AI.
    """
    # Remove Hindi (Devanagari)
    text = re.sub(r'[\u0900-\u097F]+', '', text)
    # Remove Null bytes
    text = text.replace('\x00', '')
    # Optional: Collapse excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_json_from_text(text: str):
    try:
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx == -1 or end_idx == -1: return None
        return json.loads(text[start_idx : end_idx + 1])
    except:
        return None

# ==========================================
# ENGINE 1: Backup Regex Parser
# ==========================================
def parse_with_regex(full_text: str):
    print("⚠️ AI Failed or returned empty. Switching to Regex...")
    
    # Simple regex to find "Q1." ... "a)" style format
    pattern = re.compile(r'(Q\d+\..*?)(?=Q\d+\.|$)', re.IGNORECASE)
    matches = pattern.findall(full_text)
    
    parsed_data = []
    for i, match in enumerate(matches):
        try:
            q_part = re.search(r'(Q\d+\..*?)(?=[aA]\))', match)
            q_text = q_part.group(1).strip() if q_part else f"Question {i+1}"
            
            ops = []
            for label in ['a', 'b', 'c', 'd']:
                next_label = chr(ord(label) + 1)
                if label == 'd':
                    op_pat = re.compile(rf'{label}\)(.*?)(?=Answer:|$)', re.IGNORECASE)
                else:
                    op_pat = re.compile(rf'{label}\)(.*?)(?={next_label}\))', re.IGNORECASE)
                
                op_match = op_pat.search(match)
                op_text = op_match.group(1).strip() if op_match else ""
                ops.append({"id": label, "text": op_text})

            ans_match = re.search(r'Answer:\s*\(?([a-d])\)?', match, re.IGNORECASE)
            correct = ans_match.group(1).lower() if ans_match else 'a'
            
            exp_match = re.search(r'Explanation:(.*)', match, re.IGNORECASE)
            explanation = exp_match.group(1).strip() if exp_match else "See review."

            parsed_data.append({
                "id": i+1, "text": q_text, "options": ops,
                "correct_option": correct, "explanation": explanation
            })
        except: continue
    return parsed_data

# ==========================================
# ENGINE 2: Smart AI Parser
# ==========================================
def extract_quiz_with_groq(full_text: str):
    if not GROQ_API_KEY or "PASTE_YOUR" in GROQ_API_KEY:
        return parse_with_regex(full_text)

    client = Groq(api_key=GROQ_API_KEY)
    
    # We send MORE text now (40k chars) because we cleaned it first
    prompt = f"""
    Extract ALL multiple-choice questions from the text below into a JSON Array.
    
    Rules:
    1. Output strictly a JSON list. 
    2. Extract as many questions as possible (at least 20-50).
    3. Format:
    [
      {{
        "id": 1,
        "text": "Question...",
        "options": [{{"id": "a", "text": "..."}}, {{"id": "b", "text": "..."}}],
        "correct_option": "a",
        "explanation": "..."
      }}
    ]

    Input Text:
    {full_text[:40000]} 
    """
    
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=8000  # Allow long response for many questions
        )
        
        content = completion.choices[0].message.content
        data = extract_json_from_text(content)
        
        if data and len(data) > 0:
            return data
        else:
            return parse_with_regex(full_text)

    except Exception as e:
        print(f"Groq Error: {e}")
        return parse_with_regex(full_text)

# --- Endpoints ---

@app.post("/upload", response_model=TestResponse)
async def upload_pdf(file: UploadFile):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    full_text = ""
    try:
        with pdfplumber.open(file.file) as pdf:
            # INCREASED LIMIT: Read first 50 pages
            for page in pdf.pages[:50]: 
                text = page.extract_text()
                if text:
                    # CLEAN IMMEDIATELY to save space
                    cleaned = clean_text_common(text)
                    full_text += cleaned + "\n"
    except Exception:
        pass

    if not full_text.strip():
         full_text = "Q1. Error? a) Yes b) No Answer: (a) Explanation: Empty PDF."

    quiz_data = extract_quiz_with_groq(full_text)
    
    if not quiz_data:
        quiz_data = parse_with_regex(full_text)

    # Process
    test_id = str(uuid.uuid4())
    public_questions = []
    answers_map = {}
    explanations_map = {}
    full_q_data = {}

    for i, q in enumerate(quiz_data):
        q_id = str(q.get('id', i+1))
        
        # Normalize Options
        raw_opts = q.get('options', [])
        opts = []
        if isinstance(raw_opts, list): opts = raw_opts
        elif isinstance(raw_opts, dict): opts = [{"id": k, "text": v} for k,v in raw_opts.items()]
            
        public_questions.append({
            "id": i+1, 
            "text": q.get('text', 'Question text missing'), 
            "options": opts
        })
        
        answers_map[str(i+1)] = str(q.get('correct_option', 'a')).lower().strip()
        explanations_map[str(i+1)] = q.get('explanation', 'No explanation provided.')
        
        q['options'] = opts
        full_q_data[str(i+1)] = q

    TEMP_DB[test_id] = {
        "answers": answers_map,
        "explanations": explanations_map,
        "full_data": full_q_data
    }

    return {"test_id": test_id, "questions": public_questions}

@app.post("/submit", response_model=ResultResponse)
async def submit_test(submission: Submission):
    test_id = submission.test_id
    if test_id not in TEMP_DB:
        raise HTTPException(status_code=404, detail="Test session not found")

    data = TEMP_DB[test_id]
    correct_answers = data['answers']
    full_data = data['full_data']
    
    score = 0
    total = len(correct_answers)
    details = []

    for q_id_str, correct_opt in correct_answers.items():
        user_picked = submission.user_answers.get(q_id_str)
        is_correct = False
        if user_picked:
            is_correct = str(user_picked).lower().strip() == str(correct_opt).lower().strip()
        
        if is_correct: score += 1
            
        original_q = full_data.get(q_id_str, {})
        details.append({
            "question_id": int(q_id_str),
            "question_text": original_q.get('text', ''),
            "options": original_q.get('options', []),
            "user_selected": user_picked,
            "correct_option": correct_opt,
            "is_correct": is_correct,
            "explanation": data['explanations'].get(q_id_str, "")
        })

    return {
        "score": score,
        "total": total,
        "percentage": (score / total) * 100 if total > 0 else 0,
        "details": details
    }

# ========================NANCY-------------------------
import os
import re
import json
import uuid
import time
import string
import pdfplumber
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# --- CONFIGURATION ---
# PASTE YOUR GROQ KEY BELOW
GROQ_API_KEY = "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"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-Memory DB ---
TEMP_DB = {}

# --- Models ---
class Option(BaseModel):
    id: str
    text: str

class QuestionPublic(BaseModel):
    id: int
    text: str
    options: List[Option]

class TestResponse(BaseModel):
    test_id: str
    questions: List[QuestionPublic]

class Submission(BaseModel):
    test_id: str
    user_answers: Dict[str, str]

class QuestionResult(BaseModel):
    question_id: int
    question_text: str
    options: List[Option]
    user_selected: Optional[str]
    correct_option: str
    is_correct: bool
    explanation: str

class ResultResponse(BaseModel):
    score: int
    total: int
    percentage: float
    details: List[QuestionResult]

# ==========================================
# HELPER: Pre-Cleaning
# ==========================================
def clean_text_common(text: str) -> str:
    # 1. Remove Hindi (Devanagari)
    text = re.sub(r'[\u0900-\u097F]+', '', text)
    
    # 2. Remove Specific Header/Footer Noise found in your PDF
    # We use a case-insensitive regex to catch these phrases anywhere
    noise_patterns = [
        r"SARRTHI IAS",
        r"PRELIMS NAVIGATOR PROGRAM",
        r"PNP 2026",
        r"Tes-1 \(Economy\)",
        r"www\.sarrthiias\.com",
        r"ask@sarrthi\.com",
        r"95690\s*93856",  # Handles phone number with/without spaces
        r"https://t\.me/SarrthiMentorship",
        r"Call For More Info",
        r"Difficulty:\s*Moderate", # Optional: Remove difficulty tags if they break flow
        r"Source:\s*Sarrthi IAS class notes"
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 3. Clean up excessive whitespace created by deletions
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
def extract_json_from_text(text: str):
    try:
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx == -1 or end_idx == -1: return None
        return json.loads(text[start_idx : end_idx + 1])
    except:
        return None

# ==========================================
# ENGINE 1: Backup Regex Parser (Failsafe)
# ==========================================
def parse_with_regex(full_text: str):
    print("⚠️ AI Failed or returned empty. Switching to Regex...")
    pattern = re.compile(r'(Q\d+\..*?)(?=Q\d+\.|$)', re.IGNORECASE)
    matches = pattern.findall(full_text)
    
    parsed_data = []
    for i, match in enumerate(matches):
        try:
            q_part = re.search(r'(Q\d+\..*?)(?=[aA]\))', match)
            q_text = q_part.group(1).strip() if q_part else f"Question {i+1}"
            
            ops = []
            for label in ['a', 'b', 'c', 'd']:
                next_label = chr(ord(label) + 1)
                if label == 'd':
                    op_pat = re.compile(rf'{label}\)(.*?)(?=Answer:|$)', re.IGNORECASE)
                else:
                    op_pat = re.compile(rf'{label}\)(.*?)(?={next_label}\))', re.IGNORECASE)
                
                op_match = op_pat.search(match)
                op_text = op_match.group(1).strip() if op_match else ""
                ops.append({"id": label, "text": op_text})

            ans_match = re.search(r'Answer:\s*\(?([a-d])\)?', match, re.IGNORECASE)
            correct = ans_match.group(1).lower() if ans_match else 'a'
            
            exp_match = re.search(r'Explanation:(.*)', match, re.IGNORECASE)
            explanation = exp_match.group(1).strip() if exp_match else "See review."

            parsed_data.append({
                "id": i+1, "text": q_text, "options": ops,
                "correct_option": correct, "explanation": explanation
            })
        except: continue
    return parsed_data

# ==========================================
# ENGINE 2: CHUNKED AI PARSER (The Fix)
# ==========================================
def process_chunk(client, text_chunk, chunk_index):
    """
    Sends a smaller chunk of text to the AI to prevent limits/laziness.
    """
    prompt = f"""
    Extract multiple-choice questions from this text segment into a JSON Array.
    
    Rules:
    1. Output STRICTLY VALID JSON.
    2. Format: [{{"text": "...", "options": [{{"id":"a","text":"..."}}], "correct_option": "a", "explanation": "..."}}]
    3. Ignore Hindi.
    
    Input Text (Part {chunk_index}):
    {text_chunk}
    """
    
    # We use the smaller/faster model (8b) for chunks to save rate limits
    # It is perfectly capable of extraction and 5x faster.
    model = "llama-3.1-8b-instant" 

    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0
        )
        return extract_json_from_text(completion.choices[0].message.content) or []
    except Exception as e:
        print(f"Chunk {chunk_index} failed: {e}")
        return []

def extract_quiz_with_groq_chunked(pages_text: List[str]):
    if not GROQ_API_KEY or "PASTE_YOUR" in GROQ_API_KEY:
        return parse_with_regex(" ".join(pages_text))

    client = Groq(api_key=GROQ_API_KEY)
    all_questions = []
    
    # Configuration: How many pages per chunk?
    # 5 pages is a safe balance (~2000-3000 tokens)
    PAGES_PER_CHUNK = 5 
    
    # Split pages into chunks
    chunks = []
    for i in range(0, len(pages_text), PAGES_PER_CHUNK):
        chunk_str = "\n".join(pages_text[i : i + PAGES_PER_CHUNK])
        chunks.append(chunk_str)
    
    print(f"Processing {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        print(f"Sending Chunk {i+1}/{len(chunks)} to AI...")
        
        chunk_questions = process_chunk(client, chunk, i+1)
        
        if chunk_questions:
            print(f" -> Got {len(chunk_questions)} questions from Chunk {i+1}")
            all_questions.extend(chunk_questions)
        else:
            print(f" -> Chunk {i+1} returned no data.")

        # Rate Limit Safety: Pause for 2 seconds between chunks
        time.sleep(2)

    return all_questions

# --- Endpoints ---

@app.post("/upload", response_model=TestResponse)
async def upload_pdf(file: UploadFile):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    pages_content = []
    full_text_fallback = ""
    
    try:
        with pdfplumber.open(file.file) as pdf:
            # READ ALL PAGES now (removed limit)
            for page in pdf.pages: 
                text = page.extract_text()
                if text:
                    clean = clean_text_common(text)
                    pages_content.append(clean)
                    full_text_fallback += clean + "\n"
    except Exception:
        pass

    if not pages_content:
         full_text_fallback = "Q1. Error? a) Yes b) No Answer: (a) Explanation: Empty PDF."
         pages_content = [full_text_fallback]

    # Use Chunked Processor
    quiz_data = extract_quiz_with_groq_chunked(pages_content)
    
    # Fallback if AI fails completely
    if not quiz_data or len(quiz_data) == 0:
        quiz_data = parse_with_regex(full_text_fallback)

    # Process and Renumber IDs
    test_id = str(uuid.uuid4())
    public_questions = []
    answers_map = {}
    explanations_map = {}
    full_q_data = {}

    for i, q in enumerate(quiz_data):
        # Force renumbering 1 to N
        real_id = i + 1
        
        raw_opts = q.get('options', [])
        opts = []
        if isinstance(raw_opts, list): opts = raw_opts
        elif isinstance(raw_opts, dict): opts = [{"id": k, "text": v} for k,v in raw_opts.items()]
            
        # Fallback for missing text
        q_text = q.get('text') or "Question text parsing error."
        
        public_questions.append({
            "id": real_id, 
            "text": q_text, 
            "options": opts
        })
        
        answers_map[str(real_id)] = str(q.get('correct_option', 'a')).lower().strip()
        explanations_map[str(real_id)] = q.get('explanation', 'No explanation provided.')
        
        q['options'] = opts
        full_q_data[str(real_id)] = q

    TEMP_DB[test_id] = {
        "answers": answers_map,
        "explanations": explanations_map,
        "full_data": full_q_data
    }

    return {"test_id": test_id, "questions": public_questions}

@app.post("/submit", response_model=ResultResponse)
async def submit_test(submission: Submission):
    test_id = submission.test_id
    if test_id not in TEMP_DB:
        raise HTTPException(status_code=404, detail="Test session not found")

    data = TEMP_DB[test_id]
    correct_answers = data['answers']
    full_data = data['full_data']
    
    score = 0
    total = len(correct_answers)
    details = []

    for q_id_str, correct_opt in correct_answers.items():
        user_picked = submission.user_answers.get(q_id_str)
        is_correct = False
        if user_picked:
            is_correct = str(user_picked).lower().strip() == str(correct_opt).lower().strip()
        
        if is_correct: score += 1
            
        original_q = full_data.get(q_id_str, {})
        details.append({
            "question_id": int(q_id_str),
            "question_text": original_q.get('text', ''),
            "options": original_q.get('options', []),
            "user_selected": user_picked,
            "correct_option": correct_opt,
            "is_correct": is_correct,
            "explanation": data['explanations'].get(q_id_str, "")
        })

    return {
        "score": score,
        "total": total,
        "percentage": (score / total) * 100 if total > 0 else 0,
        "details": details
    }