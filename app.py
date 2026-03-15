import os
import re
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory, send_file, make_response
from flask_cors import CORS, cross_origin
from pymongo import MongoClient
import gridfs
from bson import ObjectId
import io
import base64

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
# Enable CORS so Flutter Web can communicate with the backend
CORS(app)

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
# 🛠️ DOCUMENT VERIFICATION — OpenCV + Gemini OCR + Regex
# ==================================================
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def _normalize_dob(dob_str):
    """Normalizes any date string to DD/MM/YYYY for consistent comparison."""
    if not dob_str:
        return ""
    # Clean string
    dob_str = dob_str.strip().replace("-", "/").replace(".", "/")
    
    # Try YYYY/MM/DD -> DD/MM/YYYY
    match = re.match(r'(\d{4})/(\d{2})/(\d{2})', dob_str)
    if match:
        return f"{match.group(3)}/{match.group(2)}/{match.group(1)}"
    
    # Already DD/MM/YYYY
    match = re.match(r'(\d{2})/(\d{2})/(\d{4})', dob_str)
    if match:
        return dob_str
    
    return dob_str

def preprocess_document_image(image_bytes):
    """
    OpenCV preprocessing pipeline for better OCR clarity.
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_cv is None:
            return None, None, (0, 0)

        # Auto-rotate from EXIF using Pillow
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            from PIL import ExifTags
            for k in ExifTags.TAGS.keys():
                if ExifTags.TAGS[k] == 'Orientation':
                    break
            exif = pil_img._getexif()
            if exif and k in exif:
                rot_map = {3: cv2.ROTATE_180, 6: cv2.ROTATE_90_CLOCKWISE, 8: cv2.ROTATE_90_COUNTERCLOCKWISE}
                if exif[k] in rot_map:
                    img_cv = cv2.rotate(img_cv, rot_map[exif[k]])
        except Exception:
            pass

        # Upscale if too small
        h, w = img_cv.shape[:2]
        if max(h, w) < 1400:
            scale = 1400 / max(h, w)
            img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        original_cv = img_cv.copy()
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        return thresh, original_cv, (thresh.shape[1], thresh.shape[0])
    except Exception as e:
        print(f"Preprocess error: {e}")
        return None, None, (0, 0)

def extract_text_with# ── STAGE 2: AI ANALYST (Gemini Vision) ─────────────────────────
import json

def verify_document_with_ai(image_bytes, document_type):
    """
    Powerful AI Analyst:
    - Sends original image to Gemini 1.5 Flash.
    - Asks for structured JSON analysis.
    - Handles extraction and validation in one unified step.
    """
    if not GEMINI_API_KEY:
        return False, "Gemini API key not found", ""

    print(f"  [AI Analyst] Analyzing {document_type}...")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = (
            f"You are an expert Indian Document Analyst. Analyze this image of an Indian {document_type}.\n"
            "1. Is this a valid, authentic, and clear image of the requested document type?\n"
            "2. Extract the full Name, Date of Birth (DD/MM/YYYY), and Document Number.\n"
            "3. If it's a photo of a screen, a blurry mess, or the wrong document, set is_valid to false.\n\n"
            "Return ONLY a raw JSON strictly in this format (no markdown code blocks):\n"
            "{\"is_valid\": bool, \"name\": \"string\", \"dob\": \"DD/MM/YYYY\", \"number\": \"string\", \"reason\": \"string\"}"
        )
        
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_part = {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}}
        
        response = model.generate_content([image_part, prompt])
        raw_json = response.text.strip()
        
        # Clean markdown if AI included it
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0].strip()

        print(f"  [AI Analyst] Raw Response: {raw_json}")
        data = json.loads(raw_json)
        
        is_valid = data.get("is_valid", False)
        reason = data.get("reason", "Unknown reason")
        dob = data.get("dob", "")
        
        # Standardize DOB
        dob = _normalize_dob(dob)
        
        if is_valid:
            print(f"  ✅ [AI Analyst] Passed: {data.get('number')} | DOB: {dob}")
            return True, f"Verified: {data.get('number')}", dob
        else:
            print(f"  ❌ [AI Analyst] Failed: {reason}")
            return False, f"Invalid: {reason}", ""

    except Exception as e:
        print(f"  ⚠️ [AI Analyst] Error: {e}")
        return False, f"Analysis Error: {str(e)}", ""


def verify_document_with_gemini(image_bytes, document_type, filename="document.jpg"):
    """
    Main entry point for verification.
    """
    # Stage 1: Basic Preprocessing for quality check
    _, original_cv, _ = preprocess_document_image(image_bytes)
    if original_cv is None:
        return False, "Could not decode image", ""
        
    quality_ok, quality_msg = run_fraud_checks(original_cv)
    if not quality_ok:
        return False, quality_msg, ""

    # Stage 2: AI Analyst
    return verify_document_with_ai(image_bytes, document_type)


# ==================================================
# 👤 SAVE / UPDATE USER PROFILE
# ==================================================
@app.route('/save_profile', methods=['POST'])
@cross_origin()
def save_profile():
    try:
        email = request.form.get("email", "").strip().lower()
        if not email:
            return jsonify({"status": "error", "message": "Email is required"}), 400

        name = request.form.get("name", "")
        dob = request.form.get("dob", "")
        phone = request.form.get("phone", "")

        # Fetch existing profile to preserve data
        existing_profile = users_col.find_one({"email": email}) or {}
        
        update_doc = {
            "name": name or existing_profile.get("name", ""),
            "dob": dob or existing_profile.get("dob", ""),
            "phone": phone or existing_profile.get("phone", ""),
        }

        # --- Process Aadhaar Image ---
        aadhaar_dob = existing_profile.get("aadhaar_dob", "")
        if "aadhaar_image" in request.files:
            aadhaar_file = request.files["aadhaar_image"]
            aadhaar_bytes = aadhaar_file.read()
            
            # Clean old file
            if existing_profile.get("aadhaar_id"):
                try: fs.delete(ObjectId(existing_profile["aadhaar_id"]))
                except: pass
                
            aadhaar_id = fs.put(aadhaar_bytes, filename="aadhaar.jpg", content_type="image/jpeg")
            aadhaar_verified, _, aadhaar_dob = verify_document_with_gemini(aadhaar_bytes, "Aadhaar Card")
            
            update_doc["aadhaar_id"] = str(aadhaar_id)
            update_doc["aadhaar_verified"] = aadhaar_verified
            update_doc["aadhaar_dob"] = aadhaar_dob

        # --- Process PAN Image ---
        pan_dob = existing_profile.get("pan_dob", "")
        if "pan_image" in request.files:
            pan_file = request.files["pan_image"]
            pan_bytes = pan_file.read()
            
            # Clean old file
            if existing_profile.get("pan_id"):
                try: fs.delete(ObjectId(existing_profile["pan_id"]))
                except: pass
                
            pan_id = fs.put(pan_bytes, filename="pan.jpg", content_type="image/jpeg")
            pan_verified, _, pan_dob = verify_document_with_gemini(pan_bytes, "PAN Card")
            
            update_doc["pan_id"] = str(pan_id)
            update_doc["pan_verified"] = pan_verified
            update_doc["pan_dob"] = pan_dob

        # ---- Cross-Document DOB Matching ----
        # If we have both DOBs (either from this request or pre-existing)
        final_aadhaar_dob = update_doc.get("aadhaar_dob", existing_profile.get("aadhaar_dob", ""))
        final_pan_dob     = update_doc.get("pan_dob",     existing_profile.get("pan_dob", ""))
        
        if final_aadhaar_dob and final_pan_dob:
            if final_aadhaar_dob != final_pan_dob:
                print(f"❌ DOB Mismatch! '{final_aadhaar_dob}' vs '{final_pan_dob}'")
                update_doc["aadhaar_verified"] = False
                update_doc["pan_verified"] = False
                update_doc["verification_failure_reason"] = f"DOB mismatch: Aadhaar ({final_aadhaar_dob}) != PAN ({final_pan_dob})."
            else:
                print(f"✅ DOB Match: {final_aadhaar_dob}")
                update_doc["verification_failure_reason"] = None

        users_col.update_one({"email": email}, {"$set": update_doc}, upsert=True)
        profile = users_col.find_one({"email": email}, {"_id": 0, "password": 0})
        return jsonify({"status": "success", "profile": profile})

    except Exception as e:
        print(f"save_profile error: {e}")
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
        
        # 1. Dynamically find the best model for this particular API Key
        chat_model_name = "gemini-1.5-flash" # Default fallback
        try:
            available_models = genai.list_models()
            valid_models = [m.name for m in available_models if 'generateContent' in m.supported_generation_methods]
            print("Accessible Gemini Models:", valid_models)
            
            if valid_models:
                chat_model_name = valid_models[0] # Pick the first available
                # Prefer flash or pro if available
                for name in valid_models:
                    if 'gemini-1.5-flash' in name:
                        chat_model_name = name
                        break
        except Exception as e:
             # Catch API Key Invalid errors 
             return jsonify({"status": "error", "message": "API Key is invalid or restricted. Please create a new free API Key from Google AI Studio and update your Render Environment Variables."}), 400

        print(f"Using Google AI Model: {chat_model_name}")

        gemini_history = []
        # Inject system prompt into history for older model compatibility
        gemini_history.append({"role": "user", "parts": [SYSTEM_PROMPT]})
        gemini_history.append({"role": "model", "parts": ["[en-IN] Understood. I will act as the TrueFile AI Police Assistant, automatically detect the language, and gather the required FIR details."] })
        
        for msg in messages[:-1]: # All except the last one
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["text"]]})
            
        latest_message = messages[-1]["text"]
        
        model = genai.GenerativeModel(model_name=chat_model_name)
        
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(latest_message)
        
        return jsonify({
            "status": "success",
            "reply": response.text
        })
    except Exception as e:
        error_str = str(e)
        print(f"Chat Error: {error_str}")
        if "429" in error_str or "quota" in error_str.lower() or "limit" in error_str.lower():
            return jsonify({
                "status": "error", 
                "message": "AI Rate Limit Exceeded: You are talking too fast and hit the free API limit! Please wait 10 seconds and try your message again."
            }), 429
        return jsonify({"status": "error", "message": f"Server Error: {error_str}"}), 500

# ==================================================
# 🏠 HOME
# ==================================================
@app.route('/')
def home():
    return "Smart FIR Backend Running 🚔"

# ==================================================
# RUN
# ==================================================
if __name__ == '__main__':
    # Cloud providers like Render supply a PORT env variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

