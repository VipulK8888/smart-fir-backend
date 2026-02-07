import os
import re
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color

# MongoDB
from pymongo import MongoClient

# Translation
from googletrans import Translator

app = Flask(__name__)

# -----------------------------------------
# 🔗 MongoDB Connection
# -----------------------------------------

MONGO_URI = os.environ.get("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["fir_database"]
collection = db["confirmed_firs"]

# -----------------------------------------
# 🌐 Translator Setup
# -----------------------------------------

translator = Translator()

def translate_to_english(text):
    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        print("Translation Error:", e)
        return text   # fallback

# -----------------------------------------
# 🔍 Crime Detection
# -----------------------------------------

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

# -----------------------------------------
# 📍 Place Detection
# -----------------------------------------

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

# -----------------------------------------
# 👤 Name Detection
# -----------------------------------------

def detect_name(text):
    patterns = [
        r"my name is ([a-zA-Z ]+)",
        r"i am ([a-zA-Z ]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).title()
    return "Not Provided"

# -----------------------------------------
# 🏠 Address Detection
# -----------------------------------------

def detect_address(text):
    patterns = [
        r"i live at ([a-zA-Z0-9 ,]+)",
        r"my address is ([a-zA-Z0-9 ,]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).title()
    return "Not Provided"

# -----------------------------------------
# 🆔 FIR ID Generator
# -----------------------------------------

def generate_fir_id():
    count = collection.count_documents({}) + 1
    year = datetime.now().year
    return f"FIR-{year}-{count:04d}"

# -----------------------------------------
# 📄 PDF Generator
# -----------------------------------------

def generate_pdf(fir_text, fir_id):

    file_name = f"{fir_id}.pdf"
    c = canvas.Canvas(file_name, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(150, height - 70, "Police Department")
    c.drawString(150, height - 90, "Official FIR Report")

    y = height - 140
    c.setFont("Helvetica", 11)

    for line in fir_text.split("\n"):
        c.drawString(50, y, line.strip())
        y -= 20

    c.save()
    return file_name

# -----------------------------------------
# 🌐 API → Translate
# -----------------------------------------

@app.route('/translate', methods=['POST'])
def translate_api():

    data = request.get_json()
    original_text = data["text"]

    english_text = translate_to_english(original_text)

    return jsonify({
        "translated_text": english_text
    })

# -----------------------------------------
# 📝 Generate FIR Draft
# -----------------------------------------

@app.route('/generate_fir', methods=['POST'])
def generate_fir():

    data = request.get_json()
    description = data["english_description"]

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

    return jsonify({"fir_draft": fir_text.strip()})

# -----------------------------------------
# ✅ Confirm FIR
# -----------------------------------------

@app.route('/confirm_fir', methods=['POST'])
def confirm_fir():

    data = request.get_json()
    description = data["english_description"]

    fir_id = generate_fir_id()
    date_today = datetime.now().strftime("%d-%m-%Y")

    record = {
        "fir_id": fir_id,
        "description": description,
        "date": date_today
    }

    collection.insert_one(record)

    fir_text = f"""
FIR ID: {fir_id}
Date: {date_today}

Incident:
{description}
"""

    pdf_file = generate_pdf(fir_text, fir_id)

    return jsonify({
        "message": "FIR Confirmed",
        "fir_id": fir_id,
        "pdf": pdf_file
    })

# -----------------------------------------
# 📥 Download PDF
# -----------------------------------------

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    return send_from_directory(os.getcwd(), filename, as_attachment=True)

# -----------------------------------------
# 🏠 Home
# -----------------------------------------

@app.route('/')
def home():
    return "Smart FIR Backend Running 🚔"

# -----------------------------------------

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
