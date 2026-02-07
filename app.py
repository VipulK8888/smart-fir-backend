import os
import re
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from pymongo import MongoClient

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color

from deep_translator import GoogleTranslator

app = Flask(__name__)

# ==================================================
# 🔗 MongoDB Connection
# ==================================================

MONGO_URI = os.environ.get("MONGO_URI")

client = MongoClient(MONGO_URI)

db = client["fir_database"]
users_col = db["users"]
firs_col = db["confirmed_firs"]

# ==================================================
# 🌍 TRANSLATION → ANY LANGUAGE → ENGLISH
# ==================================================

def translate_to_english(text):

    try:
        translated = GoogleTranslator(
            source='auto',
            target='en'
        ).translate(text)

        return translated

    except:
        return text  # fallback

# ==================================================
# 🔍 NLP DETECTION
# ==================================================

CRIME_KEYWORDS = {
    "Theft": ["theft", "stolen", "steal"],
    "Robbery": ["robbery", "rob", "snatch"],
    "Assault": ["assault", "attack", "hit"],
    "Cyber Crime": ["hack", "fraud", "scam"],
    "Murder": ["murder", "killed", "dead"]
}

def detect_crime_type(text):

    text = text.lower()

    for crime, keywords in CRIME_KEYWORDS.items():
        for word in keywords:
            if word in text:
                return crime

    return "General Complaint"

def detect_name(text):

    patterns = [
        r"my name is ([a-zA-Z ]+)",
        r"i am ([a-zA-Z ]+)"
    ]

    for p in patterns:
        match = re.search(p, text.lower())
        if match:
            return match.group(1).title()

    return "Not Provided"

def detect_place(text):

    patterns = [
        r"in ([a-zA-Z ]+)",
        r"at ([a-zA-Z ]+)",
        r"near ([a-zA-Z ]+)"
    ]

    for p in patterns:
        match = re.search(p, text.lower())
        if match:
            return match.group(1).title()

    return "Not Mentioned"

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

def generate_pdf(fir_text, fir_id):

    file_name = f"{fir_id}.pdf"

    c = canvas.Canvas(file_name, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(200, height - 50,
                 "Police Department")
    c.drawString(200, height - 70,
                 "Official FIR Report")

    # FIR Text
    c.setFont("Helvetica", 11)

    y = height - 120

    for line in fir_text.split("\n"):
        c.drawString(50, y, line.strip())
        y -= 18

    # Watermark
    c.saveState()
    c.setFont("Helvetica-Bold", 60)
    c.setFillColor(Color(0.85, 0.85, 0.85, alpha=0.3))
    c.rotate(45)
    c.drawCentredString(400, 0, "DRAFT FIR")
    c.restoreState()

    c.save()

    return file_name

# ==================================================
# 👤 REGISTER
# ==================================================

@app.route('/register_user', methods=['POST'])
def register_user():

    data = request.json

    if users_col.find_one({"email": data["email"]}):
        return jsonify({"message": "User exists"})

    user = {
        "name": data["name"],
        "email": data["email"],
        "password": data["password"],
        "role": "guest"
    }

    users_col.insert_one(user)

    return jsonify({"message": "Registered"})

# ==================================================
# 🔐 LOGIN
# ==================================================

# LOGIN API
@app.route('/login', methods=['POST'])
def login():

    data = request.get_json()

    email = data['email']
    password = data['password']

    user = users_collection.find_one({
        "email": email,
        "password": password
    })

    if user:

        return jsonify({
            "status": "success",
            "name": user["name"],
            "email": user["email"],
            "role": user["role"]
        })

    else:
        return jsonify({
            "status": "failed",
            "message": "Invalid credentials"
        }), 401


# ==================================================
# 📝 GENERATE FIR DRAFT
# ==================================================

@app.route('/generate_fir', methods=['POST'])
def generate_fir():

    data = request.json
    original_text = data["description"]

    english_text = translate_to_english(original_text)

    crime = detect_crime_type(english_text)
    name = detect_name(english_text)
    place = detect_place(english_text)

    date_today = datetime.now().strftime("%d-%m-%Y")

    fir_text = f"""
FIRST INFORMATION REPORT (FIR)

Date: {date_today}
Crime Type: {crime}
Place: {place}

Complainant: {name}

Incident:
{english_text}
"""

    return jsonify({
        "fir_draft": fir_text.strip(),
        "translated_text": english_text
    })

# ==================================================
# ✅ CONFIRM FIR
# ==================================================

@app.route('/confirm_fir', methods=['POST'])
def confirm_fir():

    data = request.json

    description = data["description"]
    email = data["email"]

    fir_id = generate_fir_id()

    crime = detect_crime_type(description)
    name = detect_name(description)
    place = detect_place(description)

    date_today = datetime.now().strftime("%d-%m-%Y")

    fir_record = {
        "fir_id": fir_id,
        "email": email,
        "crime_type": crime,
        "name": name,
        "place": place,
        "description": description,
        "date": date_today
    }

    firs_col.insert_one(fir_record)

    fir_text = f"""
FIR ID: {fir_id}
Date: {date_today}
Crime: {crime}
Name: {name}
Place: {place}

Description:
{description}
"""

    pdf_file = generate_pdf(fir_text, fir_id)

    return jsonify({
        "fir_id": fir_id,
        "pdf_file": pdf_file
    })

# ==================================================
# 👮 ADMIN → ALL FIRS
# ==================================================

@app.route('/get_all_firs')
def get_all_firs():

    firs = list(
        firs_col.find({}, {"_id": 0})
    )

    return jsonify(firs)

# ==================================================
# 👤 GUEST → MY FIRS
# ==================================================

@app.route('/get_my_firs/<email>')
def get_my_firs(email):

    firs = list(
        firs_col.find(
            {"email": email},
            {"_id": 0}
        )
    )

    return jsonify(firs)

# ==================================================
# 🔎 FIR DETAIL
# ==================================================

@app.route('/get_fir/<fir_id>')
def get_fir(fir_id):

    fir = firs_col.find_one(
        {"fir_id": fir_id},
        {"_id": 0}
    )

    return jsonify(fir)

# ==================================================
# 📥 DOWNLOAD PDF
# ==================================================

@app.route('/download_pdf/<filename>')
def download_pdf(filename):

    return send_from_directory(
        directory=os.getcwd(),
        path=filename,
        as_attachment=True
    )

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
    app.run(host="0.0.0.0", port=5000)

