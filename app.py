import os
import re
import threading
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory, send_file, make_response
from flask_cors import CORS, cross_origin
from pymongo import MongoClient
import gridfs
from bson import ObjectId
import io
import base64

# NOTE: Job results are stored in MongoDB (not memory)
# so they survive Gunicorn worker restarts on Render free tier.

# Image processing for document verification pipeline
from PIL import Image, ImageEnhance, ImageFilter

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

from deep_translator import GoogleTranslator
import spacy
import subprocess
import google.generativeai as genai

# Setup Gemini API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """You are a helpful and empathetic Police Assistant helping a user draft an FIR (First Information Report).
Your goal is to gather the following details one by one from the user in a natural conversation:
1. Complainant Name
2. Age
3. Contact Number / Email
4. Incident Date and Time
5. Location of the Incident
6. Respondent Details (who committed the crime, if known)
7. The Incident Description (what happened)

Rules:
- Be empathetic and professional.
- Ask only one or two questions at a time. Do not overwhelm the user.
- CRITICAL: Automatically detect the language the user is speaking in, and ALWAYS reply in that exact same language.
- CRITICAL: Prefix EVERY single message you generate with the exact BCP-47 language code in square brackets for the language you are speaking (e.g. [en-IN], [hi-IN], [mr-IN], [bn-IN], [ta-IN], [te-IN], [gu-IN], [kn-IN], [ur-IN]). Example: '[hi-IN] नमस्ते! मैं...' or '[en-IN] Hello! I am...'.
- If the user doesn't know something (like the respondent's name), tell them it's okay and proceed.
- Once you have gathered sufficient information to write a complete FIR, output a final message starting exactly with '[FIR_COMPLETE]' followed immediately by '[en-IN]' (or whichever language) and then a detailed narrative in that language containing all the information collected. The narrative MUST EVENTUALLY BE ONLY A SIMPLE PARAGRAPH OR SET OF PARAGRAPHS describing the incident. DO NOT write a formal letter (i.e., do not include "To, The Police Inspector," subject lines, dates at the top, or signatures at the bottom). Just write the storytelling paragraph(s) outlining exactly what happened. Do not append any other conversational text after the narrative. Let the narrative be the entire unadulterated payload after the '[FIR_COMPLETE]' keyword.
"""

# Load SpaCy model, download if missing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading SpaCy NLP model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# ── CORS — allow Flutter Web (any origin) to POST multipart/form-data ──
CORS(app,
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization", "Accept"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=False)

# Handle OPTIONS preflight for every route explicitly
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        res = app.make_default_options_response()
        res.headers["Access-Control-Allow-Origin"]  = "*"
        res.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        res.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept"
        return res

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept"
    return response

# ==================================================
# 🔗 MongoDB Connection
# ==================================================
# Use a default fallback URI for local development if the environment variable isn't set
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")

try:
    client = MongoClient(MONGO_URI)
    db = client["fir_database"]
    users_col = db["users"]
    firs_col = db["confirmed_firs"]
    fs = gridfs.GridFS(db)
    print("Successfully connected to MongoDB and GridFS.")
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")

# ==================================================
# 🌍 TRANSLATION FUNCTION (SAFE)
# ==================================================
def translate_to_english(text):
    try:
        if not text.strip():
            return ""
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated
    except Exception as e:
        print("Translation Error:", e)
        return text   # fallback

# ==================================================
# 🌍 TRANSLATE API
# ==================================================
@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Missing request data"}), 400
            
        text = data.get("text", "")
        english_text = translate_to_english(text)

        return jsonify({
            "status": "success",
            "translated_text": english_text
        })
    except Exception as e:
        print("Translate API Error:", e)
        # Using "error" instead of "failed" to match the Dart frontend checks
        return jsonify({
            "status": "error",
            "message": "Translation failed",
            "translated_text": ""
        }), 500

# ==================================================
# 🔍 NLP DETECTION
# ==================================================
CRIME_KEYWORDS = {
    "Theft/Burglary": ["theft", "stolen", "steal", "chori", "robbery", "snatch", "loot", "pickpocket", "burgle", "rob"],
    "Assault/Violence": ["assault", "attack", "hit", "beat", "punch", "stab", "violence", "murder", "kill", "dead", "fight"],
    "Cyber Crime": ["hack", "fraud", "scam", "online", "phishing", "bank", "otp"],
    "Harassment/Sexual Offense": ["harass", "rape", "molest", "touch", "outrage", "stalk", "eve"],
    "Kidnapping/Abduction": ["kidnap", "abduct", "missing", "ransom"]
}

def detect_crime_type(text):
    text = text.lower()
    for crime, keywords in CRIME_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return crime
    return "General Complaint"

def extract_entities(text):
    doc = nlp(text)
    names = [ent.text.title() for ent in doc.ents if ent.label_ == "PERSON"]
    places = [ent.text.title() for ent in doc.ents if ent.label_ in ["GPE", "LOC", "FAC"]]
    
    # Exclude basic stop words that spacy sometimes misidentifies
    stop_words = ["i", "me", "my", "he", "she", "they", "we", "us", "him", "her", "unknown", "some"]
    names = [n for n in names if n.lower() not in stop_words]
    
    name = names[0] if len(names) > 0 else "Unknown"
    respondent = names[1] if len(names) > 1 else "Unknown"
    place = ", ".join(set(places)) if places else "Unknown"

    contact_match = re.search(r'\b\d{10}\b', text)
    contact = contact_match.group(0) if contact_match else "Unknown"

    age_match = re.search(r'\b(\d{1,3})\s*(?:years|yrs)\s*(?:old|of age)?\b|\bage(?:d)?\s*(?:is)?\s*(\d{1,3})\b', text, re.IGNORECASE)
    age = "Unknown"
    if age_match:
        age = age_match.group(1) or age_match.group(2) or "Unknown"

    height_match = re.search(r'\b(\d{1,3}(?:\.\d{1,2})?)\s*(cm|m|feet|ft|inches|in)\b', text, re.IGNORECASE)
    height = f"{height_match.group(1)} {height_match.group(2)}" if height_match else "Unknown"
    
    return {
        "name": name,
        "respondent": respondent,
        "place": place,
        "address": place,
        "contact": contact,
        "age": age,
        "height": height,
        "demographic": "Unknown"
    }

# ==================================================
# 🆔 FIR ID GENERATOR
# ==================================================
def generate_fir_id():
    count = firs_col.count_documents({}) + 1
    year = datetime.now().year
    return f"FIR-{year}-{count:04d}"

# ==================================================
# 📄 PDF GENERATION
# ==================================================
def add_watermark(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 45)
    canvas.setFillColor(colors.Color(0.85, 0.85, 0.85, alpha=0.3))
    canvas.translate(300, 400)
    canvas.rotate(45)
    canvas.drawCentredString(0, 0, "SYSTEM GENERATED DRAFT")
    canvas.restoreState()

def generate_pdf(fir_data, fir_id):
    os.makedirs("pdfs", exist_ok=True)
    file_name = os.path.join("pdfs", f"{fir_id}.pdf")
    
    doc = SimpleDocTemplate(file_name, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=40, bottomMargin=30)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        alignment=1, # Center
        fontSize=18,
        spaceAfter=5
    )
    sub_title_style = ParagraphStyle(
        name='SubTitle', 
        parent=styles['Normal'], 
        alignment=1, 
        fontSize=9,
        leading=11,
        spaceAfter=20
    )
    normal_style = styles['Normal']
    
    # 1. Header
    header_style = ParagraphStyle(
        name='HeaderStyle',
        parent=styles['Normal'],
        alignment=0, # Left aligned
        fontSize=12,
        spaceAfter=15
    )
    elements.append(Paragraph("<b>Police Department<br/>Official FIR Report</b>", header_style))

    elements.append(Paragraph("<b>INCIDENT REPORT</b>", title_style))
    elements.append(Paragraph("", sub_title_style))

    # 2. Main Large Table
    table_data = []

    # Row 1: Date
    table_data.append([Paragraph(f"<b>Date:</b> {fir_data.get('date', datetime.now().strftime('%d %B %Y'))}", normal_style), ""])
    
    # Row 2: Location
    table_data.append([Paragraph(f"<b>Location:</b> {fir_data.get('place', 'Unknown')}", normal_style), ""])

    # Row 3: Complainant Details (Split columns)
    table_data.append([
        Paragraph("<b>Complainant Details:</b><br/>" + fir_data.get('name', 'Unknown').split()[0], normal_style),
        Paragraph("<b>Last Name:</b><br/>" + (fir_data.get('name', 'Unknown').split()[-1] if len(fir_data.get('name', 'Unknown').split()) > 1 else ""), normal_style)
    ])

    # Row 4: Respondent Details
    table_data.append([Paragraph(f"<b>Respondent Details:</b><br/>{fir_data.get('respondent', 'Unknown')}", normal_style), ""])

    # Row 5: Address (Split)
    table_data.append([
        Paragraph("<b>Address:</b><br/>" + fir_data.get('address', 'Unknown'), normal_style),
        Paragraph("<b>Contact number:</b><br/>" + fir_data.get('contact', fir_data.get('email', 'Unknown')), normal_style)
    ])

    # Row 6: Height/Age
    table_data.append([
        Paragraph("<b>Complainant Height:</b> " + fir_data.get('height', 'Unknown'), normal_style),
        Paragraph("<b>Age:</b> " + fir_data.get('age', 'Unknown'), normal_style)
    ])

    # Row 7: Pen Name
    table_data.append([
         Paragraph("<b>Pen Name:</b><br/>Complainant details", normal_style),
         Paragraph("<b>Community:</b>", normal_style)
    ])

    # Row 8: Incident Description
    desc_para = Paragraph("<b>Incident Description:</b><br/>" + fir_data.get('description', 'No description provided.'), normal_style)
    table_data.append([desc_para, ""])

    # Removed redundant Complainant Information rows here


    # Construct Table
    t = Table(table_data, colWidths=[260, 260])

    # Add complex spanning and grid styling to match the image exactly
    grid_style = [
        # Box around the entire table
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        
        # Span columns for full-width rows
        ('SPAN', (0,0), (1,0)), # Date
        ('SPAN', (0,1), (1,1)), # Location
        ('SPAN', (0,3), (1,3)), # Respondent
        ('SPAN', (0,7), (1,7)), # Description (make it tall)
        
        # Padding
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]

    t.setStyle(TableStyle(grid_style))
    elements.append(t)
    
    # 3. Footer (Signatures)
    elements.append(Spacer(1, 40))
    elements.append(Paragraph("Investigating Officer Signature: ______________________", normal_style))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Station Seal: ______________________", normal_style))
    
    doc.build(elements, onFirstPage=add_watermark, onLaterPages=add_watermark)
    return file_name

# ==================================================
# 👤 REGISTER USER
# ==================================================
@app.route('/register_user', methods=['POST'])
def register_user():
    try:
        data = request.get_json()
        if not data or not all(k in data for k in ("name", "email", "password")):
             return jsonify({"status": "error", "message": "Missing name, email, or password"}), 400

        if users_col.find_one({"email": data["email"]}):
            return jsonify({
                "status": "error", # Changed from 'failed' to matched frontend logic
                "message": "User already exists"
            })

        user = {
            "name": data["name"],
            "email": data["email"],
            "password": data["password"],
            "role": "user" # Matches flutter logic
        }

        users_col.insert_one(user)

        return jsonify({
            "status": "success",
            "message": "Registered successfully"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==================================================
# 🔐 LOGIN
# ==================================================
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data or not all(k in data for k in ("email", "password")):
             return jsonify({"status": "error", "message": "Missing email or password"}), 400

        email = data.get("email")
        password = data.get("password")

        user = users_col.find_one({
            "email": email,
            "password": password
        })

        if user:
            return jsonify({
                "status": "success",
                "name": user["name"],
                "email": user["email"],
                "role": user.get("role", "user")
            })
        else:
            return jsonify({
                "status": "error", # Changed from 'failed'
                "message": "Invalid credentials"
            }), 401
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==================================================
# 👮 CREATE ADMIN (RUN ONCE)
# ==================================================
@app.route('/create_admin')
def create_admin():
    admin = {
        "name": "Admin",
        "email": "admin@police.com", # Changed to match typical testing emails
        "password": "admin", # Simple password for testing
        "role": "admin"
    }

    if not users_col.find_one({"email": admin["email"]}):
        users_col.insert_one(admin)
        return "Admin Created Successfully"
    
    return "Admin already exists"

# ==================================================
# 📝 GENERATE FIR DRAFT
# ==================================================
@app.route('/generate_fir', methods=['POST'])
def generate_fir():
    try:
        data = request.get_json()
        if not data or "description" not in data:
             return jsonify({"status": "error", "message": "Missing description"}), 400
             
        original_text = data["description"]

        english_text = translate_to_english(original_text)

        crime = detect_crime_type(english_text)
        entities = extract_entities(english_text)
        name = entities.get("name", "Unknown")
        respondent = entities.get("respondent", "Unknown")
        place = entities.get("place", "Unknown")
        address = entities.get("address", "Unknown")
        contact = entities.get("contact", "Unknown")
        age = entities.get("age", "Unknown")
        height = entities.get("height", "Unknown")
        demographic = entities.get("demographic", "Unknown")

        date_today = datetime.now().strftime("%d-%m-%Y")

        fir_text = f"""FIRST INFORMATION REPORT (FIR)

Date: {date_today}
Crime Type: {crime}
Place: {place}

Complainant: {name}

Incident:
{english_text}
"""

        return jsonify({
            "status": "success", # Added standard status
            "fir_draft": fir_text.strip(),
            "fir_data": {
                "date": date_today,
                "crime_type": crime,
                "place": place,
                "name": name,
                "description": english_text,
                "respondent": respondent,
                "address": address,
                "contact": contact,
                "age": age,
                "height": height,
                "demographic": demographic
            },
            "translated_text": english_text
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==================================================
# ✅ CONFIRM FIR
# ==================================================
@app.route('/confirm_fir', methods=['POST'])
def confirm_fir():
    try:
        # Support both JSON (original) and Form Data (with file)
        if request.is_json:
            data = request.get_json()
            description = data.get("description")
            email = data.get("email", "guest")
        else:
            description = request.form.get("description")
            email = request.form.get("email", "guest")
            
        if not description:
             return jsonify({"status": "error", "message": "Missing description"}), 400

        # Optional Evidence Processing
        evidence_id = None
        if 'evidence' in request.files:
            file = request.files['evidence']
            if file and file.filename != '':
                # Save binary to GridFS
                evidence_id = fs.put(file, filename=file.filename, content_type=file.content_type)
                evidence_id = str(evidence_id)

        fir_id = generate_fir_id()

        crime = detect_crime_type(description)
        entities = extract_entities(description)
        name = entities.get("name", "Unknown")
        respondent = entities.get("respondent", "Unknown")
        place = entities.get("place", "Unknown")
        address = entities.get("address", "Unknown")
        contact = entities.get("contact", "Unknown")
        age = entities.get("age", "Unknown")
        height = entities.get("height", "Unknown")
        demographic = entities.get("demographic", "Unknown")

        date_today = datetime.now().strftime("%d-%m-%Y")

        fir_record = {
            "fir_id": fir_id,
            "email": email,
            "crime_type": crime,
            "name": name,
            "respondent": respondent,
            "place": place,
            "address": address,
            "contact": contact,
            "age": age,
            "height": height,
            "demographic": demographic,
            "description": description,
            "evidence_id": evidence_id,
            "date": date_today,
            "status": "Pending" 
        }

        firs_col.insert_one(fir_record)

        pdf_file = generate_pdf(fir_record, fir_id)

        return jsonify({
            "status": "success",
            "fir_id": fir_id,
            "pdf_file": pdf_file
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==================================================
# 👮 ADMIN → UPDATE FIR STATUS
# ==================================================
@app.route('/update_fir_status', methods=['POST'])
def update_fir_status():
    try:
        data = request.get_json()
        if not data or not all(k in data for k in ("fir_id", "status")):
            return jsonify({"status": "error", "message": "Missing fir_id or status"}), 400
            
        fir_id = data["fir_id"]
        status = data["status"] # 'Approved' or 'Rejected'
        
        result = firs_col.update_one(
            {"fir_id": fir_id},
            {"$set": {"status": status}}
        )
        
        if result.modified_count > 0:
            return jsonify({"status": "success", "message": f"FIR marked as {status}"})
        else:
            return jsonify({"status": "error", "message": "FIR not found or status already matches"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==================================================
# 👮 ADMIN → ALL FIRS
# ==================================================
@app.route('/get_all_firs')
def get_all_firs():
    try:
        firs = list(firs_col.find({}, {"_id": 0}).sort("_id", -1)) # Add sort by newest
        return jsonify(firs)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==================================================
# 👤 GUEST → MY FIRS
# ==================================================
@app.route('/get_my_firs/<email>')
def get_my_firs(email):
    try:
        firs = list(firs_col.find({"email": email}, {"_id": 0}).sort("_id", -1))
        return jsonify(firs)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==================================================
# 🔎 FIR DETAIL
# ==================================================
@app.route('/get_fir/<fir_id>')
def get_fir(fir_id):
    try:
        fir = firs_col.find_one({"fir_id": fir_id}, {"_id": 0})
        if fir:
             return jsonify(fir)
        return jsonify({"status": "error", "message": "unfound"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==================================================
# 🛠️ DOCUMENT VERIFICATION — OpenCV + Regex + Gemini AI
# 3-Stage Pipeline (EasyOCR removed — too heavy for Render free tier):
#   Stage 1 → OpenCV Image Enhancement + Quality Check
#   Stage 2 → Gemini AI extracts text + validates document
#   Stage 3 → Regex validates Gemini's extracted number / DOB
# ==================================================
import cv2
import numpy as np
import json
from PIL import Image, ImageEnhance

# ── HELPERS ──────────────────────────────────────────────────────

def _normalize_dob(dob_str):
    """Normalises any date string (including month names) to DD/MM/YYYY."""
    if not dob_str:
        return ""
    import dateutil.parser as dparser
    try:
        parsed_date = dparser.parse(dob_str, dayfirst=True)
        return parsed_date.strftime("%d/%m/%Y")
    except Exception:
        dob_str = dob_str.strip().replace("-", "/").replace(".", "/")
        match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', dob_str)
        if match:
            return f"{match.group(1).zfill(2)}/{match.group(2).zfill(2)}/{match.group(3)}"
        return dob_str

# Regex patterns for post-AI validation
_AADHAAR_RE = re.compile(r'\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b')
_PAN_RE     = re.compile(r'\b([A-Z]{5}[0-9]{4}[A-Z])\b')
_DOB_RE     = re.compile(
    r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})'
    r'|\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b',
    re.IGNORECASE,
)

def _regex_validate(number, dob, document_type):
    """
    Cross-checks Gemini's extracted number and DOB against Regex patterns.
    Returns (number, dob) — corrected or original.
    """
    if number:
        if "aadhaar" in document_type.lower():
            m = _AADHAAR_RE.search(number)
            if m:
                number = re.sub(r'[\s\-]', '', m.group(1))
                print(f"  [Regex] Aadhaar number validated: {number}")
            else:
                print(f"  [Regex] ⚠️ Aadhaar number format mismatch: {number}")
        elif "pan" in document_type.lower():
            m = _PAN_RE.search(number.upper())
            if m:
                number = m.group(1)
                print(f"  [Regex] PAN number validated: {number}")
            else:
                print(f"  [Regex] ⚠️ PAN number format mismatch: {number}")

    if dob:
        dob_match = _DOB_RE.search(dob)
        if dob_match:
            raw = dob_match.group(1) or dob_match.group(2)
            dob = _normalize_dob(raw)
            print(f"  [Regex] DOB validated: {dob}")
        else:
            # Try normalising as-is
            dob = _normalize_dob(dob)
            print(f"  [Regex] DOB normalised: {dob}")

    return number, dob

# ── STAGE 1: OPENCV IMAGE ENHANCEMENT ───────────────────────────

def preprocess_document_image(image_bytes):
    """
    OpenCV pipeline:
    - EXIF auto-rotation
    - Upscaling for low-res images
    - Resize large images to max 1600px (saves Gemini bandwidth)
    - Grayscale → Bilateral denoising → Adaptive thresholding
    Returns (enhanced_bytes, original_cv, (width, height))
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_cv is None:
            return image_bytes, None, (0, 0)

        # Auto-rotate using EXIF orientation tag
        try:
            from PIL import ExifTags
            pil_img = Image.open(io.BytesIO(image_bytes))
            orient_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            exif = pil_img._getexif() if hasattr(pil_img, "_getexif") else None
            if exif and orient_key and orient_key in exif:
                rot_map = {
                    3: cv2.ROTATE_180,
                    6: cv2.ROTATE_90_CLOCKWISE,
                    8: cv2.ROTATE_90_COUNTERCLOCKWISE,
                }
                if exif[orient_key] in rot_map:
                    img_cv = cv2.rotate(img_cv, rot_map[exif[orient_key]])
        except Exception:
            pass

        h, w = img_cv.shape[:2]

        # Upscale if too small
        if max(h, w) < 1000:
            scale  = 1000 / max(h, w)
            img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_CUBIC)
            h, w = img_cv.shape[:2]

        # Downscale if too large (saves Gemini API bandwidth & speeds up response)
        if max(h, w) > 1600:
            scale  = 1600 / max(h, w)
            img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_AREA)
            h, w = img_cv.shape[:2]

        original_cv = img_cv.copy()

        # Encode back to JPEG bytes for Gemini
        _, buf = cv2.imencode(".jpg", img_cv, [cv2.IMWRITE_JPEG_QUALITY, 90])
        enhanced_bytes = buf.tobytes()

        return enhanced_bytes, original_cv, (w, h)
    except Exception as e:
        print(f"  [OpenCV] Preprocess error: {e}")
        return image_bytes, None, (0, 0)

# ── STAGE 1b: QUALITY / FRAUD CHECK ─────────────────────────────

def run_fraud_checks(original_img):
    """Rejects obviously blurry images before spending Gemini API credits."""
    if original_img is None:
        return True, "Check skipped"
    gray          = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 5: # Drastically lowered to 5 to avoid false positives on valid cards
        return False, (
            f"Image too blurry (sharpness score: {laplacian_var:.0f}). "
            "Please retake with better lighting."
        )
    return True, "Quality OK"

# ── STAGE 2: GEMINI AI EXTRACTION + VALIDATION ───────────────────

def verify_document_with_ocr(image_bytes, document_type, filename="document.jpg"):
    """
    NEW ARCHITECTURE: Bypasses Google Gemini entirely!
    Stage 1 → OpenCV enhancement + quality check
    Stage 2 → Sends image to OCR.space (Specialized FREE Cloud OCR)
    Stage 3 → High-Accuracy Python Regex extracts Aadhaar/PAN and DOB from the raw text layout
    Returns (is_valid: bool, message: str, dob: str)
    """
    import requests
    
    print(f"\n{'='*55}")
    print(f"  📄 Verifying: {document_type} [{filename}]")
    print(f"{'='*55}")

    # Stage 1 — OpenCV Quality Check
    enhanced_bytes, original_cv, dimensions = preprocess_document_image(image_bytes)
    if original_cv is None:
        print("  [Stage 1] ⚠️ OpenCV failed — using raw bytes")
        enhanced_bytes = image_bytes
    else:
        print(f"  [Stage 1] OpenCV ✅ — size: {dimensions}")

    quality_ok, quality_msg = run_fraud_checks(original_cv)
    if not quality_ok:
        return False, quality_msg, ""
    print(f"  [Stage 1] Quality check ✅")

    # Stage 2 — OCR.space Text Extraction
    print(f"  [OCR.space] Reading text from {document_type}…")
    b64_image = base64.b64encode(enhanced_bytes).decode("utf-8")
    b64_string = f"data:image/jpeg;base64,{b64_image}"
    
    # Use environment key or the default free-tier fallback key 'helloworld'
    api_key = os.environ.get("OCR_SPACE_API_KEY", "helloworld")
    
    payload = {
        'apikey': api_key,
        'base64Image': b64_string,
        'language': 'eng',
        'isOverlayRequired': False,
        'OCREngine': 2, # Engine 2 is much more accurate for reading numbers on ID cards!
        'scale': True
    }
    
    try:
        response = requests.post('https://api.ocr.space/parse/image', data=payload, timeout=25)
        result = response.json()
        
        if result.get('IsErroredOnProcessing'):
            err = result.get('ErrorMessage', ['Unknown Error'])[0]
            print(f"  ❌ [OCR.space] Error: {err}")
            return False, f"OCR Error: {err}", ""
            
        parsed_results = result.get('ParsedResults', [])
        if not parsed_results:
            print("  ❌ [OCR.space] No text extracted")
            return False, "Could not extract any text from the provided image.", ""
            
        raw_text = parsed_results[0].get('ParsedText', '')
        print("  [OCR] Raw Text Extracted from ID:")
        print(f"    {repr(raw_text)}")
        
        # Stage 3 — Deterministic Regex Extraction
        number_found = ""
        dob_found = ""
        
        if "aadhaar" in document_type.lower():
            m = _AADHAAR_RE.search(raw_text)
            if m: number_found = re.sub(r'[\s\-]', '', m.group(1))
        elif "pan" in document_type.lower():
            m = _PAN_RE.search(raw_text.upper())
            if m: number_found = m.group(1)
            
        dob_match = _DOB_RE.search(raw_text)
        if dob_match:
            raw_dob = dob_match.group(1) or dob_match.group(2)
            dob_found = _normalize_dob(raw_dob)
            
        if number_found:
            msg = f"Verified: {number_found}"
            print(f"  ✅ [Regex] Passed! ID: {number_found} | DOB: {dob_found}")
            print(f"{'='*55}\n")
            return True, msg, dob_found
        else:
            print(f"  ❌ [Regex] Failed: Could not locate valid {document_type} pattern in text.")
            print(f"{'='*55}\n")
            return False, f"Invalid: Could not find valid {document_type} format in image", ""
            
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️ [OCR.space] Connection Error: {e}")
        return False, "OCR Server connection failed. Ensure you have internet access.", ""
    except Exception as e:
        print(f"  ⚠️ [OCR.space] Unexpected Error: {e}")
        return False, str(e), ""


# ==================================================
# 👤 SAVE / UPDATE USER PROFILE (Async — avoids Render 30s timeout)
# Job results stored in MongoDB so they survive worker restarts.
# Flutter should poll /verification_status/<job_id> every 4 seconds.
# ==================================================
@app.route('/save_profile', methods=['POST'])
@cross_origin()
def save_profile():
    try:
        email = request.form.get("email", "").strip().lower()
        if not email:
            return jsonify({"status": "error", "message": "Email is required"}), 400

        # Read ALL file bytes NOW — request context closes after this function returns
        aadhaar_bytes = request.files["aadhaar_image"].read() if "aadhaar_image" in request.files else None
        pan_bytes     = request.files["pan_image"].read()     if "pan_image"     in request.files else None
        name          = request.form.get("name", "")
        dob           = request.form.get("dob", "")
        phone         = request.form.get("phone", "")

        job_id = email  # use email as unique job key

        # Store job status in MongoDB — survives Gunicorn worker restarts
        db["verification_jobs"].update_one(
            {"job_id": job_id},
            {"$set": {"job_id": job_id, "status": "processing", "updated_at": datetime.utcnow()}},
            upsert=True
        )

        def run_verification():
            try:
                print(f"🔄 [Thread] Starting verification for {email}")
                existing_profile = users_col.find_one({"email": email}) or {}
                update_doc = {
                    "name":  name  or existing_profile.get("name", ""),
                    "dob":   dob   or existing_profile.get("dob", ""),
                    "phone": phone or existing_profile.get("phone", ""),
                }

                # --- Process Aadhaar Image ---
                if aadhaar_bytes:
                    if existing_profile.get("aadhaar_id"):
                        try: fs.delete(ObjectId(existing_profile["aadhaar_id"]))
                        except: pass
                    aadhaar_id = fs.put(aadhaar_bytes, filename="aadhaar.jpg", content_type="image/jpeg")
                    aadhaar_verified, _, aadhaar_dob = verify_document_with_ocr(aadhaar_bytes, "Aadhaar Card")
                    update_doc["aadhaar_id"]       = str(aadhaar_id)
                    update_doc["aadhaar_verified"] = aadhaar_verified
                    update_doc["aadhaar_dob"]      = aadhaar_dob

                # --- Process PAN Image ---
                if pan_bytes:
                    if existing_profile.get("pan_id"):
                        try: fs.delete(ObjectId(existing_profile["pan_id"]))
                        except: pass
                    pan_id = fs.put(pan_bytes, filename="pan.jpg", content_type="image/jpeg")
                    pan_verified, _, pan_dob = verify_document_with_ocr(pan_bytes, "PAN Card")
                    update_doc["pan_id"]       = str(pan_id)
                    update_doc["pan_verified"] = pan_verified
                    update_doc["pan_dob"]      = pan_dob

                # ---- Cross-Document DOB Matching ----
                final_aadhaar_dob = update_doc.get("aadhaar_dob", existing_profile.get("aadhaar_dob", ""))
                final_pan_dob     = update_doc.get("pan_dob",     existing_profile.get("pan_dob", ""))
                if final_aadhaar_dob and final_pan_dob:
                    if final_aadhaar_dob != final_pan_dob:
                        print(f"❌ DOB Mismatch! '{final_aadhaar_dob}' vs '{final_pan_dob}'")
                        update_doc["aadhaar_verified"] = False
                        update_doc["pan_verified"]     = False
                        update_doc["verification_failure_reason"] = (
                            f"DOB mismatch: Aadhaar ({final_aadhaar_dob}) != PAN ({final_pan_dob})."
                        )
                    else:
                        print(f"✅ DOB Match: {final_aadhaar_dob}")
                        update_doc["verification_failure_reason"] = None

                users_col.update_one({"email": email}, {"$set": update_doc}, upsert=True)
                profile = users_col.find_one({"email": email}, {"_id": 0, "password": 0})

                # Save result to MongoDB
                db["verification_jobs"].update_one(
                    {"job_id": job_id},
                    {"$set": {
                        "status": "done",
                        "profile": profile,
                        "updated_at": datetime.utcnow()
                    }},
                    upsert=True
                )
                print(f"✅ Verification job complete for {email}")

            except Exception as e:
                print(f"❌ Verification job error for {email}: {e}")
                db["verification_jobs"].update_one(
                    {"job_id": job_id},
                    {"$set": {
                        "status": "error",
                        "message": str(e),
                        "updated_at": datetime.utcnow()
                    }},
                    upsert=True
                )

        # Launch in background thread — returns 202 instantly to Flutter
        threading.Thread(target=run_verification, daemon=True).start()
        return jsonify({"status": "processing", "job_id": job_id}), 202

    except Exception as e:
        print(f"save_profile error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ==================================================
# 🔄 VERIFICATION STATUS POLL ENDPOINT
# Flutter calls this every 4s after receiving "processing"
# Results fetched from MongoDB — survives worker restarts
# ==================================================
@app.route('/verification_status/<job_id>', methods=['GET'])
@cross_origin()
def verification_status(job_id):
    try:
        job = db["verification_jobs"].find_one({"job_id": job_id}, {"_id": 0})
        if not job:
            return jsonify({"status": "processing"}), 200  # Still starting up
        # Return "success" instead of "done" for Flutter compatibility
        if job.get("status") == "done":
            job["status"] = "success"
        return jsonify(job)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ==================================================
# 👤 GET USER PROFILE
# ==================================================
@app.route('/get_profile/<email>')
@cross_origin()
def get_profile(email):
    try:
        profile = users_col.find_one({"email": email}, {"_id": 0, "password": 0})
        if profile:
            return jsonify({"status": "success", "profile": profile})
        return jsonify({"status": "not_found", "profile": {}})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==================================================
# 📷 SERVE EVIDENCE FILE (GRIDFS)
# ==================================================
@app.route('/evidence/<evidence_id>')
def serve_evidence(evidence_id):
    try:
        if not evidence_id or evidence_id == "None":
            return "No evidence attached", 404
            
        file_data = fs.get(ObjectId(evidence_id))
        response = make_response(file_data.read())
        response.headers.set('Content-Type', file_data.content_type)
        response.headers.set('Content-Disposition', 'inline', filename=file_data.filename)
        return response
    except Exception as e:
        print(f"Error serving evidence {evidence_id}: {e}")
        return "Evidence not found", 404

# ==================================================
# 📥 SERVE OR REGENERATE PDF
# ==================================================
@app.route('/pdfs/<filename>')
@cross_origin()
def serve_pdf(filename):
    try:
        # Check if the file exists locally first
        if os.path.exists(os.path.join("pdfs", filename)):
            return send_from_directory('pdfs', filename, as_attachment=True)
            
        # If it doesn't exist (due to server restart), regenerate it from MongoDB
        fir_id = filename.replace(".pdf", "")
        fir = firs_col.find_one({"fir_id": fir_id})
        
        if not fir:
            return jsonify({"status": "error", "message": "FIR not found"}), 404
            
        # Regenerate the PDF physically
        generate_pdf(fir, fir_id)
        
        # Serve the newly generated file
        if os.path.exists(os.path.join("pdfs", filename)):
            return send_from_directory('pdfs', filename, as_attachment=True)
        else:
            return jsonify({"status": "error", "message": "Failed to generate PDF on the fly"}), 500
            
    except Exception as e:
        return jsonify({"status": "error", "message": f"Server Error: {str(e)}"}), 500
# ==================================================
# 🤖 CHATBOT API (GEMINI GUIDED FIR)
# ==================================================
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or "messages" not in data:
            return jsonify({"status": "error", "message": "Missing messages array"}), 400
            
        if not GEMINI_API_KEY:
             return jsonify({"status": "error", "message": "Backend is missing GEMINI_API_KEY environment variable. Chatbot is offline."}), 500
             
        messages = data["messages"]
        
        gemini_history = []
        # Inject system prompt into history for older model compatibility
        gemini_history.append({"role": "user", "parts": [SYSTEM_PROMPT]})
        gemini_history.append({"role": "model", "parts": ["[en-IN] Understood. I will act as the TrueFile AI Police Assistant, automatically detect the language, and gather the required FIR details."] })
        
        for msg in messages[:-1]: # All except the last one
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["text"]]})
            
        latest_message = messages[-1]["text"]
        
        # Smart AI Model Hopping Architecture
        # If the primary model hits a limit, it silently cascades to backup models 
        # to ensure the user never gets a 429 API Limit Exceeded block.
        models_to_run = [
            "gemini-2.0-flash",        # Primary High-Speed Model
            "gemini-1.5-flash",        # Backup 1
            "gemini-1.5-flash-8b"      # Backup 2
        ]
        
        reply_text = None
        last_error = ""

        for chat_model_name in models_to_run:
            try:
                print(f"[Chatbot] Trying Google AI Model: {chat_model_name}...")
                model = genai.GenerativeModel(model_name=chat_model_name)
                chat_session = model.start_chat(history=gemini_history)
                response = chat_session.send_message(latest_message)
                reply_text = response.text
                print(f"[Chatbot] Success using {chat_model_name}!")
                break # Exit the loop if it worked!
            except Exception as e:
                error_str = str(e)
                print(f"[Chatbot] Model {chat_model_name} failed: {error_str}")
                last_error = error_str
                # If it's a structural error (not a quota error) fail immediately
                is_quota_error = any(k in error_str.lower() for k in ["429", "quota", "rate", "limit", "resource_exhausted", "503", "500", "overloaded"])
                if not is_quota_error:
                    break
        
        if reply_text is not None:
            return jsonify({
                "status": "success",
                "reply": reply_text
            })
        else:
            # All models failed or exhausted
            if "429" in last_error or "quota" in last_error.lower() or "limit" in last_error.lower():
                return jsonify({
                    "status": "error", 
                    "message": "AI Rate Limit Exceeded: Google's free quotas are temporarily maxed out across all backup models. Please wait exactly 1 minute."
                }), 429
            return jsonify({"status": "error", "message": f"Server Error: {last_error}"}), 500
    except Exception as general_e:
        return jsonify({"status": "error", "message": f"Server Error: {str(general_e)}"}), 500

# ==================================================
# 🏠 HOME
# ==================================================
@app.route('/')
def home():
    return "Smart FIR Backend Running 🚔"

# ==================================================
# 🔍 VERSION CHECK — confirms which code is deployed
# ==================================================
@app.route('/version')
def version():
    return jsonify({
        "version": "3.0",
        "gemini_model": "gemini-2.0-flash",
        "job_store": "mongodb",
        "status": "ok"
    })

# ==================================================
# 🧹 CLEAR STALE VERIFICATION JOB (call once if stuck)
# Usage: GET /clear_job/<email>
# ==================================================
@app.route('/clear_job/<job_id>', methods=['GET'])
@cross_origin()
def clear_job(job_id):
    try:
        result = db["verification_jobs"].delete_one({"job_id": job_id})
        if result.deleted_count > 0:
            return jsonify({"status": "success", "message": f"Job cleared for {job_id}"})
        return jsonify({"status": "success", "message": "No job found to clear"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==================================================
# 🧹 CLEAR STALE PROFILE VERIFICATION DATA
# Usage: GET /clear_profile_verification/<email>
# Resets aadhaar/pan verification fields in user profile
# ==================================================
@app.route('/clear_profile_verification/<email>', methods=['GET'])
@cross_origin()
def clear_profile_verification(email):
    try:
        result = users_col.update_one(
            {"email": email},
            {"$unset": {
                "aadhaar_verified": "",
                "aadhaar_dob": "",
                "aadhaar_message": "",
                "aadhaar_id": "",
                "pan_verified": "",
                "pan_dob": "",
                "pan_message": "",
                "pan_id": "",
                "verification_failure_reason": ""
            }}
        )
        # Also clear the job
        db["verification_jobs"].delete_one({"job_id": email})
        return jsonify({
            "status": "success",
            "message": f"Profile verification data cleared for {email}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==================================================
# RUN
# ==================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
