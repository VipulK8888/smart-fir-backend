import os
import re
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
from pymongo import MongoClient

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

from deep_translator import GoogleTranslator
import spacy
import subprocess

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
    print("Successfully connected to MongoDB.")
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
    names = [n for n in names if n.lower() not in ["i", "me", "my", "he", "she", "they", "we", "us", "him", "her"]]
    
    return {
        "name": names[0] if names else "Not Provided",
        "place": ", ".join(set(places)) if places else "Not Mentioned"
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
    canvas.setFont("Helvetica-Bold", 60)
    canvas.setFillColor(colors.Color(0.85, 0.85, 0.85, alpha=0.3))
    canvas.translate(300, 400)
    canvas.rotate(45)
    canvas.drawCentredString(0, 0, "CONFIDENTIAL")
    canvas.restoreState()

def generate_pdf(fir_data, fir_id):
    os.makedirs("pdfs", exist_ok=True)
    file_name = os.path.join("pdfs", f"{fir_id}.pdf")
    
    doc = SimpleDocTemplate(file_name, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        alignment=1, # Center
        fontSize=14,
        spaceAfter=10
    )
    normal_style = styles['Normal']
    
    # Header
    elements.append(Paragraph("<b>FIRST INFORMATION REPORT</b>", title_style))
    elements.append(Paragraph("(Under Section 154 Cr.P.C.)", ParagraphStyle(name='SubTitle', parent=styles['Normal'], alignment=1, spaceAfter=20)))
    
    # Information Table
    data1 = [
        ["1. District / Police Station:", "Virtual Police Station", "Year:", str(datetime.now().year)],
        ["   FIR No.:", fir_id, "Date:", fir_data.get('date', datetime.now().strftime('%d-%m-%Y'))],
    ]
    t1 = Table(data1, colWidths=[150, 150, 80, 150])
    t1.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(t1)
    elements.append(Spacer(1, 10))
    
    data2 = [
        ["2. Acts & Sections:", fir_data.get('crime_type', 'General Offense')],
        ["3. (a) Occurrence of Offense:", "Date: Unknown   Time: Unknown "],
        ["   (b) Information received at P.S.:", f"Date: {fir_data.get('date', '-')} "],
        ["4. Type of Information:", "Written / Typed (Online Submission)"],
    ]
    t2 = Table(data2, colWidths=[180, 350])
    t2.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 10))
    
    data3 = [
        ["5. Place of Occurrence:", fir_data.get('place', 'Unknown')],
        ["6. Complainant / Informant:", ""],
        ["   (a) Name:", fir_data.get('name', 'Unknown')],
        ["   (b) Contact / Email:", fir_data.get('email', 'Unknown')],
        ["7. Details of accused:", "As described in the incident"],
        ["8. Reasons for delay:", "N/A"],
        ["9. Properties Stolen:", "N/A"],
    ]
    
    t3 = Table(data3, colWidths=[180, 350])
    t3.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(t3)
    elements.append(Spacer(1, 10))
    
    # Description
    elements.append(Paragraph("<b>10. First Information contents (Description of Incident):</b>", styles['Heading3']))
    elements.append(Spacer(1, 5))
    
    incident_text = fir_data.get('description', '')
    for line in incident_text.split('\n'):
        if line.strip():
            elements.append(Paragraph(line.strip(), normal_style))
            elements.append(Spacer(1, 2))
    
    elements.append(Spacer(1, 40))
    
    # Signatures
    sig_data = [
        ["-----------------------------------------", "-----------------------------------------"],
        ["Signature / Thumb impression", "Signature of Officer-in-Charge"],
        ["of the Complainant / Informant", "Police Station"],
    ]
    t_sig = Table(sig_data, colWidths=[265, 265])
    t_sig.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(t_sig)
    
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
        name = entities["name"]
        place = entities["place"]

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
        data = request.get_json()
        if not data or "description" not in data:
             return jsonify({"status": "error", "message": "Missing description"}), 400

        description = data["description"]
        email = data.get("email", "guest")

        fir_id = generate_fir_id()

        crime = detect_crime_type(description)
        entities = extract_entities(description)
        name = entities["name"]
        place = entities["place"]

        date_today = datetime.now().strftime("%d-%m-%Y")

        fir_record = {
            "fir_id": fir_id,
            "email": email,
            "crime_type": crime,
            "name": name,
            "place": place,
            "description": description,
            "date": date_today,
            "status": "Pending" # Defaults to Pending now
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
        firs = list(firs_col.find({}, {"_id": 0}).sort("date", -1)) # Add sort by newest
        return jsonify(firs)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==================================================
# 👤 GUEST → MY FIRS
# ==================================================
@app.route('/get_my_firs/<email>')
def get_my_firs(email):
    try:
        firs = list(firs_col.find({"email": email}, {"_id": 0}).sort("date", -1))
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
# 📥 SERVE OR REGENERATE PDF
# ==================================================
@app.route('/pdfs/<filename>')
@cross_origin()
def serve_pdf(filename):
    try:
        # Check if the file exists locally first
        if os.path.exists(os.path.join("pdfs", filename)):
            return send_from_directory('pdfs', filename)
            
        # If it doesn't exist (due to server restart), regenerate it from MongoDB
        fir_id = filename.replace(".pdf", "")
        fir = firs_col.find_one({"fir_id": fir_id})
        
        if not fir:
            return jsonify({"status": "error", "message": "FIR not found"}), 404
            
        # Regenerate the PDF physically
        generate_pdf(fir, fir_id)
        
        # Serve the newly generated file
        if os.path.exists(os.path.join("pdfs", filename)):
            return send_from_directory('pdfs', filename)
        else:
            return jsonify({"status": "error", "message": "Failed to generate PDF on the fly"}), 500
            
    except Exception as e:
        return jsonify({"status": "error", "message": f"Server Error: {str(e)}"}), 500
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

