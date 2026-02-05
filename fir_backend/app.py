import pandas as pd
import os
import re
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)

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
    text = text.lower()

    patterns = [
        r"near ([a-zA-Z ]+)",
        r"at ([a-zA-Z ]+)",
        r"in ([a-zA-Z ]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().title()

    return "Not Mentioned"


# --------------------------------------------------
# NAME DETECTION
# --------------------------------------------------
def detect_name(text):
    text = text.lower()

    patterns = [
        r"my name is ([a-zA-Z ]+)",
        r"i am ([a-zA-Z ]+)",
        r"this is ([a-zA-Z ]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).title()

    return "Not Provided"


# --------------------------------------------------
# ADDRESS DETECTION
# --------------------------------------------------
def detect_address(text):
    text = text.lower()

    patterns = [
        r"i live at ([a-zA-Z0-9 ,]+)",
        r"my address is ([a-zA-Z0-9 ,]+)",
        r"resident of ([a-zA-Z0-9 ,]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().title()

    return "Not Provided"


# --------------------------------------------------
# FIR ID GENERATION
# --------------------------------------------------
def generate_fir_id():

    file_name = "confirmed_firs.xlsx"

    if os.path.exists(file_name):
        df = pd.read_excel(file_name)
        count = len(df) + 1
    else:
        count = 1

    year = datetime.now().year

    return f"FIR-{year}-{count:04d}"


# --------------------------------------------------
# EXCEL STORAGE
# --------------------------------------------------
def save_to_excel(fir_id, date, crime_type,
                  place, name, address, description):

    file_name = "confirmed_firs.xlsx"

    data = {
        "FIR ID": [fir_id],
        "Date": [date],
        "Crime Type": [crime_type],
        "Place": [place],
        "Name": [name],
        "Address": [address],
        "Incident Description": [description]
    }

    df_new = pd.DataFrame(data)

    if os.path.exists(file_name):
        df_existing = pd.read_excel(file_name)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_excel(file_name, index=False, engine="openpyxl")


# --------------------------------------------------
# PDF GENERATION (LOGO + SIGNATURE + WATERMARK)
# --------------------------------------------------
def generate_pdf(fir_text, fir_id):

    file_name = f"{fir_id}.pdf"

    c = canvas.Canvas(file_name, pagesize=A4)
    width, height = A4

    # ---------------- WATERMARK ----------------
    c.saveState()

    c.setFont("Helvetica-Bold", 60)
    c.setFillGray(0.9, 0.3)  # Transparent grey

    c.translate(width/2, height/2)
    c.rotate(45)

    c.drawCentredString(0, 0, "DRAFT FIR")

    c.restoreState()

    # ---------------- LOGO ----------------
    logo_path = "police_logo.png"

    if os.path.exists(logo_path):
        c.drawImage(logo_path,
                    50, height - 100,
                    width=60, height=60)

    # ---------------- HEADER ----------------
    c.setFont("Helvetica-Bold", 14)
    c.drawString(150, height - 70, "Police Department")
    c.drawString(150, height - 90, "Official FIR Report")

    # ---------------- FIR TEXT ----------------
    c.setFont("Helvetica", 11)

    y = height - 140

    for line in fir_text.split("\n"):
        c.drawString(50, y, line.strip())
        y -= 20

        if y < 100:
            c.showPage()
            y = height - 50

    # ---------------- SIGNATURE ----------------
    c.drawString(50, 120,
                 "Investigating Officer Signature: __________")

    c.drawString(50, 90,
                 "Station Seal: __________")

    c.save()

    return file_name


# --------------------------------------------------
# API → GENERATE FIR
# --------------------------------------------------
@app.route('/generate_fir', methods=['POST'])
def generate_fir():

    data = request.get_json(force=True, silent=True)

    if not data or "description" not in data:
        return jsonify({"error": "Invalid request"}), 400

    description = data["description"]

    crime_type = detect_crime_type(description)
    place = detect_place(description)
    name = detect_name(description)
    address = detect_address(description)
    date_today = datetime.now().strftime("%d-%m-%Y")

    fir_draft = f"""
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

    return jsonify({"fir_draft": fir_draft.strip()})


# --------------------------------------------------
# API → CONFIRM FIR
# --------------------------------------------------
@app.route('/confirm_fir', methods=['POST'])
def confirm_fir():

    data = request.get_json(force=True, silent=True)

    if not data or "description" not in data:
        return jsonify({"error": "Invalid request"}), 400

    description = data["description"]

    crime_type = detect_crime_type(description)
    place = detect_place(description)
    name = detect_name(description)
    address = detect_address(description)
    date_today = datetime.now().strftime("%d-%m-%Y")

    fir_id = generate_fir_id()

    fir_draft = f"""
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

    save_to_excel(fir_id, date_today,
                  crime_type, place,
                  name, address, description)

    pdf_file = generate_pdf(fir_draft, fir_id)

    return jsonify({
        "message": "FIR Confirmed & Saved",
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
# RUN SERVER
# --------------------------------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
