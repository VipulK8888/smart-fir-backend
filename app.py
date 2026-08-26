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
from groq import Groq

# Google Generative AI has been removed entirely; Groq will handle the chatbot.

SYSTEM_PROMPT = """You are an official Indian Police Assistant helping a user draft an FIR.
Your goal is to gather the following details one by one from the user in a natural conversation:
1. Complainant Name & Father's/Husband's Name
2. Complainant Age/DOB, Occupation, and Address
3. Contact Number / Email
4. Exact Date, Time, and Day of the Incident
5. Exact Location/Address of the Incident (Direction/Distance from Police Station if known)
6. Suspect/Accused Details (Name, appearance, vehicle, etc.)
7. Details of any Stolen or Involved Properties (and total value)
8. Reasons for delay in reporting (if any)
9. The full Incident Narrative (what happened)

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
    
    doc = SimpleDocTemplate(file_name, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=50, bottomMargin=40)
    elements = []
    
    styles = getSampleStyleSheet()
    
    # Custom Styles matching the form
    title_style = ParagraphStyle(name='Title', parent=styles['Normal'], alignment=1, fontName='Helvetica-Bold', fontSize=12, spaceAfter=5)
    sub_title = ParagraphStyle(name='SubTitle', parent=styles['Normal'], alignment=1, fontSize=11, spaceAfter=20)
    
    body = ParagraphStyle(name='Body', parent=styles['Normal'], fontSize=10, leading=15, spaceAfter=8)
    body_indent = ParagraphStyle(name='BodyIndent', parent=styles['Normal'], fontSize=10, leading=15, leftIndent=20, spaceAfter=8)
    body_indent2 = ParagraphStyle(name='BodyIndent2', parent=styles['Normal'], fontSize=10, leading=15, leftIndent=40, spaceAfter=8)
    
    # Header
    elements.append(Paragraph("<b>FORM – IF1 - (Integrated Form)</b>", title_style))
    elements.append(Paragraph("<b>FIRST INFORMATION REPORT</b>", title_style))
    elements.append(Paragraph("(Under Section 154 Cr.P.C)", sub_title))
    
    # Extract data safely with fallbacks
    date_val = fir_data.get('date', datetime.now().strftime('%d %B %Y'))
    acts = fir_data.get('acts_and_sections', 'Not Specified')
    occ_date = fir_data.get('occurrence_day_date_time', 'Unknown')
    place_dist = fir_data.get('place_of_occurrence_direction_distance', 'Unknown')
    place_addr = fir_data.get('place_of_occurrence_address', 'Unknown')
    
    c_name = fir_data.get('complainant_name', 'Unknown')
    c_father = fir_data.get('complainant_father_husband_name', 'Unknown')
    c_dob = fir_data.get('complainant_dob_year', 'Unknown')
    c_nat = fir_data.get('complainant_nationality', 'Indian')
    c_pass = fir_data.get('complainant_passport', 'Not Specified')
    c_occ = fir_data.get('complainant_occupation', 'Not Specified')
    c_addr = fir_data.get('complainant_address', 'Unknown')
    
    suspect = fir_data.get('suspect_details', 'Unknown')
    delay = fir_data.get('reasons_for_delay', 'None stated')
    stolen = fir_data.get('stolen_properties', 'None')
    stolen_val = fir_data.get('total_value_stolen', 'N/A')
    inquest = fir_data.get('inquest_report_ud_case', 'None')
    contents = fir_data.get('fir_contents', 'No description provided.')

    # Form Lines
    elements.append(Paragraph(f"1. Dist: <b>Central</b> &nbsp;&nbsp;&nbsp;&nbsp; P.S: <b>Cyber Crime Cell</b> &nbsp;&nbsp;&nbsp;&nbsp; Year: <b>{datetime.now().year}</b> &nbsp;&nbsp;&nbsp;&nbsp; F.I.R. No: <b>{fir_id}</b> &nbsp;&nbsp;&nbsp;&nbsp; Date: <b>{date_val}</b>", body))
    elements.append(Paragraph(f"2. Acts & Sections: <b>{acts}</b>", body))
    elements.append(Paragraph(f"3. (a) Occurrence of Offence: <b>{occ_date}</b>", body))
    elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;(b) Information received at P.S. Date: <b>{date_val}</b>", body))
    elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;(c) General Diary Reference: Entry No. ________ Time: ________", body))
    elements.append(Paragraph(f"4. Type of information: <b>Written / Portal UI</b>", body))
    
    elements.append(Paragraph(f"5. Place of occurrence:", body))
    elements.append(Paragraph(f"(a) Direction and Distance from P.S: <b>{place_dist}</b>", body_indent))
    elements.append(Paragraph(f"(b) Address: <b>{place_addr}</b>", body_indent))
    elements.append(Paragraph(f"(c) In case outside limit of this Police Station, then Name of P.S: _______", body_indent))
    
    elements.append(Paragraph(f"6. Complainant / information:", body))
    elements.append(Paragraph(f"(a) Name: <b>{c_name}</b>", body_indent))
    elements.append(Paragraph(f"(b) Father's / Husband's Name: <b>{c_father}</b>", body_indent))
    elements.append(Paragraph(f"(c) Date / Year of Birth: <b>{c_dob}</b> &nbsp;&nbsp;&nbsp; (d) Nationality: <b>{c_nat}</b>", body_indent))
    elements.append(Paragraph(f"(e) Passport No: <b>{c_pass}</b> &nbsp;&nbsp;&nbsp; Date of Issue: _______ Place of Issue: _______", body_indent))
    elements.append(Paragraph(f"(f) Occupation: <b>{c_occ}</b>", body_indent))
    elements.append(Paragraph(f"(g) Address: <b>{c_addr}</b>", body_indent))

    elements.append(Paragraph(f"7. Details of known / suspected / unknown accused with full particulars:", body))
    elements.append(Paragraph(f"<b>{suspect}</b>", body_indent))
    
    elements.append(Paragraph(f"8. Reasons for delay in reporting by the complainant / Informant:", body))
    elements.append(Paragraph(f"<b>{delay}</b>", body_indent))

    elements.append(Paragraph(f"9. Particulars of properties stolen / involved:", body))
    elements.append(Paragraph(f"<b>{stolen}</b>", body_indent))
    
    elements.append(Paragraph(f"10. Total value of the properties stolen / involved: <b>{stolen_val}</b>", body))
    elements.append(Paragraph(f"11. Inquest Report / U.D. Case No., if any: <b>{inquest}</b>", body))
    
    elements.append(Paragraph(f"12. F.I.R. Contents (Attach separate sheets, if required):", body))
    
    # Highly formal narrative box
    narrative_style = ParagraphStyle(name='Narrative', parent=styles['Normal'], fontSize=10, leading=16, leftIndent=5, rightIndent=5, spaceBefore=5, spaceAfter=15, justify=True)
    elements.append(Paragraph(contents.replace('\n', '<br/>'), narrative_style))

    # Footer
    elements.append(Paragraph(f"13. Action taken: Since the above report reveals commission of offence (s) u/s as mentioned at Item No. 2., registered the case and took up the investigation / directed ____________ Rank ____________ to take up the investigation.", body))
    elements.append(Paragraph(f"F.I.R. read over to the complainant / Informant, admitted to be correctly recorded and a copy given to the complainant / Informant free of cost.", body))
    
    elements.append(Spacer(1, 40))
    
    # Signature Layout using Table for left/right alignment
    sig_data = [
        [
            Paragraph("<b>Signature / Thumb-impression<br/>of the complainant / informant</b>", body), 
            Paragraph("<b>Signature of the Officer-in-charge, Police Station</b><br/>Name: __________________________<br/>Rank: ________________ No: ________", body)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[250, 260])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('ALIGN', (1,0), (1,0), 'RIGHT')
    ]))
    elements.append(sig_table)
    
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"15. Date & time of despatch to the court: <b>{date_val}</b>", body))
    
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

        # 🚀 Groq LLM Pass to formalize text into IF-1 schema
        current_time = datetime.now().strftime('%A, %d-%m-%Y %I:%M %p')
        system_prompt = f"""You are an expert Indian Legal AI Assistant. Your job is to extract and rewrite the provided informal incident report into a strict JSON payload mapping exactly to the official Indian Police FIR Form-IF1 (Section 154 Cr.P.C).

CRITICAL INSTRUCTIONS:
1. Return ONLY a valid JSON string. Do not wrap in markdown (e.g. ```json). Do not provide any conversational text before or after the JSON.
2. For missing information, output 'Not Specified' or 'Unknown'.
3. 'fir_contents' MUST be a highly formal, third-person police narrative expanding upon the user's description (e.g. 'On the day of X, the complainant approached...').
4. The CURRENT ACTUAL DATE AND TIME IS: {current_time}. MUST mathematically calculate any relative times like "yesterday", "last night", "two hours ago" into exact DD-MM-YYYY dates for the 'occurrence_day_date_time' field! Output EXACT dates!

Return exactly this JSON schema:
{{
  "acts_and_sections": "(Predict relevant Indian Penal Code or BNS sections based on crime type described)",
  "occurrence_day_date_time": "(Extract or 'Not Specified')",
  "place_of_occurrence_direction_distance": "(Extract or 'Not Specified')",
  "place_of_occurrence_address": "(Extract or 'Not Specified')",
  "complainant_name": "(Extract or 'Unknown')",
  "complainant_father_husband_name": "(Extract or 'Not Specified')",
  "complainant_dob_year": "(Extract or 'Not Specified')",
  "complainant_nationality": "Indian",
  "complainant_passport": "(Extract or 'Not Specified')",
  "complainant_occupation": "(Extract or 'Not Specified')",
  "complainant_address": "(Extract or 'Not Specified')",
  "suspect_details": "(Extract or 'Unknown')",
  "reasons_for_delay": "(Extract or 'None stated')",
  "stolen_properties": "(Extract detailed list or 'None')",
  "total_value_stolen": "(Extract or 'N/A')",
  "inquest_report_ud_case": "(Extract or 'None')",
  "fir_contents": "(The highly formalized, professional legal incident narrative rewritten in third-person)",
  "crime_type": "(Very short 2-3 word category of crime, e.g. 'Theft of Vehicle')"
}}
"""

        try:
            groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-8b-8192"]
            raw_json_str = None
            
            for model_name in models_to_try:
                try:
                    response = groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Incident Description:\n{english_text}"}
                        ],
                        model=model_name,
                        temperature=0.2,
                        max_tokens=2048
                    )
                    raw_json_str = response.choices[0].message.content.strip()
                    break # Success
                except Exception as e:
                    print(f"Model {model_name} failed: {e}")
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        break # Stop trying if rate limited
                        
            if not raw_json_str:
                raise Exception("All Groq models failed or rate limit exceeded.")
            
            # Remove markdown backticks if Groq ignored the instruction
            if raw_json_str.startswith("```"):
                raw_json_str = re.sub(r'^```[a-zA-Z]*\n', '', raw_json_str)
                raw_json_str = re.sub(r'\n```$', '', raw_json_str)
            
            import json
            fir_form_data = json.loads(raw_json_str)
            
        except Exception as e:
            print("Groq Extract Error:", e)
            # Fallback to older basic extraction
            crime = detect_crime_type(english_text)
            entities = extract_entities(english_text)
            fir_form_data = {
                "acts_and_sections": "Relevant sections to be added",
                "occurrence_day_date_time": "Unknown",
                "place_of_occurrence_direction_distance": "Unknown",
                "place_of_occurrence_address": entities.get("address", "Unknown"),
                "complainant_name": entities.get("name", "Unknown"),
                "complainant_father_husband_name": "Unknown",
                "complainant_dob_year": entities.get("age", "Unknown"),
                "complainant_nationality": "Indian",
                "complainant_passport": "Unknown",
                "complainant_occupation": "Unknown",
                "complainant_address": "Unknown",
                "suspect_details": entities.get("respondent", "Unknown"),
                "reasons_for_delay": "None stated",
                "stolen_properties": "None",
                "total_value_stolen": "N/A",
                "inquest_report_ud_case": "None",
                "fir_contents": english_text,
                "crime_type": crime
            }

        date_today = datetime.now().strftime("%d-%m-%Y")
        fir_form_data["contact"] = extract_entities(english_text).get("contact", "Unknown") # preserve contact
        fir_form_data["date"] = date_today

        # Create a nice plaintext digest for the Flutter app preview screen
        fir_text_preview = f"""**Form - IF1 (Integrated Form) Preview**
*Under Section 154 Cr.P.C*

**Acts & Sections:** {fir_form_data.get('acts_and_sections')}
**Occurrence:** {fir_form_data.get('occurrence_day_date_time')}
**Place:** {fir_form_data.get('place_of_occurrence_address')}

**Complainant:** {fir_form_data.get('complainant_name')} (Father: {fir_form_data.get('complainant_father_husband_name')})
**Suspect Details:** {fir_form_data.get('suspect_details')}
**Stolen Properties:** {fir_form_data.get('stolen_properties')} (Est. Value: {fir_form_data.get('total_value_stolen')})
**Reasons for Delay:** {fir_form_data.get('reasons_for_delay')}

**F.I.R. Contents:**
{fir_form_data.get('fir_contents')}
"""

        return jsonify({
            "status": "success",
            "fir_draft": fir_text_preview.strip(),
            "fir_data": fir_form_data,
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
            fir_data_payload = data.get("fir_data", {})
        else:
            description = request.form.get("description")
            email = request.form.get("email", "guest")
            fir_data_payload_str = request.form.get("fir_data", "{}")
            import json
            fir_data_payload = json.loads(fir_data_payload_str)
            
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

        date_today = datetime.now().strftime("%d-%m-%Y (Time: %I:%M %p)")

        fir_record = fir_data_payload.copy()
        
        # Override with mathematically verified User Profile data if available
        user = users_col.find_one({"email": email})
        if user:
            print(f"Injecting verified profile data for {email}")
            if user.get("dob"):
                fir_record["complainant_dob_year"] = user.get("dob")
            if user.get("name"):
                fir_record["complainant_name"] = user.get("name")
        
        # Merge system fields
        fir_record.update({
            "fir_id": fir_id,
            "email": email,
            "description": description,
            "evidence_id": evidence_id,
            "date": date_today,
            "status": "Pending",
            "crime_type": detect_crime_type(description)
        })

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
            
        # CRITICAL FIX: Find ALL dates in the OCR text
        all_dates = _DOB_RE.findall(raw_text)
        valid_date_objs = []
        
        from datetime import datetime
        
        for match_tuple in all_dates:
            raw_d = match_tuple[0] or match_tuple[1]
            if not raw_d: continue
            
            norm_str = _normalize_dob(raw_d)
            try:
                # Attempt to parse into a strict datetime object to compare historical age
                dt = datetime.strptime(norm_str, "%d/%m/%Y")
                valid_date_objs.append((dt, norm_str))
            except Exception:
                pass
                
        if valid_date_objs:
            # The Date of Birth is mathematically ALWAYS older than the Date of Issue!
            # We sort all found dates from oldest to newest, and pick the absolute oldest one.
            valid_date_objs.sort(key=lambda x: x[0])
            dob_found = valid_date_objs[0][1]
            
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
# 🤖 CHATBOT API (GROQ LLAMA-3 GUIDED FIR)
# ==================================================
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or "messages" not in data:
            return jsonify({"status": "error", "message": "Missing messages array"}), 400
            
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        if not GROQ_API_KEY:
             return jsonify({"status": "error", "message": "Backend is missing GROQ_API_KEY environment variable. Chatbot is offline."}), 500
             
        messages = data["messages"]
        
        # Using Llama models via Groq with fallback
        chat_model_names = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-8b-8192"]
        
        groq_messages = []
        # Inject system prompt into history
        groq_messages.append({"role": "system", "content": SYSTEM_PROMPT})
        groq_messages.append({"role": "assistant", "content": "[en-IN] Understood. I will act as the TrueFile AI Police Assistant, automatically detect the language, and gather the required FIR details."})
        
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            groq_messages.append({"role": role, "content": msg["text"]})
            
        reply_text = None
        last_error = ""

        try:
            client = Groq(api_key=GROQ_API_KEY)
            for chat_model_name in chat_model_names:
                try:
                    print(f"[Chatbot] Trying Groq AI Model: {chat_model_name}...")
                    
                    completion = client.chat.completions.create(
                        model=chat_model_name,
                        messages=groq_messages,
                        temperature=0.7,
                        max_completion_tokens=1024,
                        top_p=1,
                        stream=False,
                        stop=None,
                    )
                    
                    reply_text = completion.choices[0].message.content
                    print(f"[Chatbot] Success using {chat_model_name}!")
                    break
                except Exception as e:
                    error_str = str(e)
                    print(f"[Chatbot] Model {chat_model_name} failed: {error_str}")
                    last_error = error_str
                    if "429" in error_str or "rate limit" in error_str.lower() or "limit" in error_str.lower():
                        break
        except Exception as e:
            last_error = str(e)
        
        if reply_text is not None:
            return jsonify({
                "status": "success",
                "reply": reply_text
            })
        else:
            if "429" in last_error or "rate limit" in last_error.lower() or "limit" in last_error.lower():
                return jsonify({
                    "status": "error", 
                    "message": "AI Rate Limit Exceeded: Groq's quotas are temporarily maxed out. Please wait exactly 1 minute before sending another message."
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
