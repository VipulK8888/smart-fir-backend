import os
import re
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color

# MongoDB
from pymongo import MongoClient

# 🌐 Translation
from deep_translator import GoogleTranslator

app = Flask(__name__)

# --------------------------------------------------
# 🔗 MongoDB CONNECTION
# --------------------------------------------------

MONGO_URI = os.environ.get("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["fir_database"]
collection = db["confirmed_firs"]

# --------------------------------------------------
# 🌐 TRANSLATOR SETUP
# --------------------------------------------------

def translate_to_english(text):
    try:
        translated = GoogleTranslator(
            source='auto',
            target='en'
        ).translate(text)

        return translated

    except Exception as e:
        print("Translation Error:", e)
        return text

# --------------------------------------------------
# NLP → CRIME DETECTION
# --------------------------------------------------

CRIME_KEYWORDS = {
    "Theft": ["theft", "stolen", "steal"],
    "Robbery": ["robbery", "rob", "snatch"],
    "Assault": ["assault", "attack", "hit"],
    "Cyber Crime": ["hack", "fraud", "scam", "online"],
    "Murder": ["murder", "killed", "dead"]
}

def detect_crime_type(text):
    text = text.lower()
    for crime, keywords in CRIME_KEYWORDS.items():
        for word in keywords:
            if word in text:
                return crime
    return "General Complaint"

# --------------------------------------------------
# PLACE DETECTION
# --------------------------------------------------

def detect_place(text):
    patterns = [
        r"near ([a-zA-Z ]+)",
        r"at ([a-zA-Z ]+)",
        r"in ([a-zA-Z ]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).title()

    return "Not Mentioned"

# --------------------------------------------------
# NAME DETECTION
# --------------------------------------------------

def detect_name(text):
    patterns = [
        r"my name is ([a-zA-Z ]+)",
        r"i am ([a-zA-Z ]+)",
        r"this is ([a-zA-Z ]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).title()

    return "Not Provided"

# --------------------------------------------------
# ADDRESS DETECTION
# --------------------------------------------------

def detect_address(text):
    patterns = [
        r"i live at ([a-zA-Z0-9 ,]+)",
        r"my address is ([a-zA-Z0-9 ,]+)",
        r"resident of ([a-zA-Z0-9 ,]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).title()

    return "Not Provided"

# --------------------------------------------------
# FIR ID GENERATOR
# --------------------------------------------------

def generate_fir_id():
    count = collection.count_documents({}) + 1
    year = datetime.now().year
    return f"FIR-{year}-{count:04d}"

# --------------------------------------------------
# EXCEL BACKUP
# --------------------------------------------------

def save_to_excel(data):

    file_name = "confirmed_firs.xlsx"
    df_new = pd.DataFrame([data])

    if os.path.exists(file_name):
        df_existing = pd.read_excel(file_name)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_excel(file_name, index=False)

# --------------------------------------------------
# PDF GENERATION
# --------------------------------------------------

def generate_pdf(fir_text, fir_id, watermark=None):

    file_name = f"{fir_id}.pdf"

    c = canvas.Canvas(file_name, pagesize=A4)
    width, height = A4

    logo_path = "police_logo.png"
    if os.path.exists(logo_path):
        c.drawImage(logo_path, 50, height - 100, width=60, height=60)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(150, height - 70, "Police Department")
    c.drawString(150, height - 90, "Official FIR Report")

    c.setFont("Helvetica", 11)
    y = height - 140

    for line in fir_text.split("\n"):
        c.drawString(50, y, line.strip())
        y -= 20

    if watermark:
        c.saveState()
        c.setFont("Helvetica-Bold", 60)
        c.setFillColor(Color(0.8, 0.8, 0.8, alpha=0.3))
        c.rotate(45)
        c.drawCentredString(300, 0, watermark)
        c.restoreState()

    c.save()
    return file_name

# --------------------------------------------------
# API → GENERATE FIR
# --------------------------------------------------

@app.route('/generate_fir', methods=['POST'])
def generate_fir():

    data = request.get_json()
    description_original = data["description"]

    # 🌐 Translate
    description = translate_to_english(description_original)

    crime_type = detect_crime_type(description)
    place = detect_place(description)
    name = detect_name(description)
    address = detect_address(description)

    date_today = datetime.now().strftime("%d-%m-%Y")

    fir_text = f"""
FIRST INFORMATION REPORT (FIR)

Date: {date_today}
Crime Type: {crime_type}
Place of Incident: {place}

Complainant Details:
Name: {name}
Address: {address}

Incident Details:
{description}
"""

    return jsonify({
        "fir_draft": fir_text.strip(),
        "translated_text": description
    })

# --------------------------------------------------
# API → CONFIRM FIR
# --------------------------------------------------

@app.route('/confirm_fir', methods=['POST'])
def confirm_fir():

    data = request.get_json()
    description_original = data["description"]

    # 🌐 Translate
    description = translate_to_english(description_original)

    crime_type = detect_crime_type(description)
    place = detect_place(description)
    name = detect_name(description)
    address = detect_address(description)

    date_today = datetime.now().strftime("%d-%m-%Y")
    fir_id = generate_fir_id()

    fir_record = {
        "fir_id": fir_id,
        "date": date_today,
        "crime_type": crime_type,
        "place": place,
        "name": name,
        "address": address,
        "description_original": description_original,
        "description_translated": description,
        "status": "Confirmed"
    }

    collection.insert_one(fir_record)
    save_to_excel(fir_record)

    fir_text = f"""
FIRST INFORMATION REPORT (FIR)

FIR ID: {fir_id}
Date: {date_today}
Crime Type: {crime_type}
Place of Incident: {place}

Complainant Details:
Name: {name}
Address: {address}

Incident Details:
{description}
"""

    pdf_file = generate_pdf(fir_text, fir_id)

    return jsonify({
        "message": "FIR Confirmed",
        "fir_id": fir_id,
        "pdf_file": pdf_file
    })

# --------------------------------------------------
# DOWNLOAD PDF
# --------------------------------------------------

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    return send_from_directory(
        directory=os.getcwd(),
        path=filename,
        as_attachment=True
    )

# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route('/')
def home():
    return "Smart FIR Backend Running 🚔"

# --------------------------------------------------

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

