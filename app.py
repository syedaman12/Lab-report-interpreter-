import os, io, json, re
from datetime import datetime
from flask import Flask, jsonify, request, render_template, send_file
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ---------- OpenRouter client ----------
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

DATA_FILE = "lab_reports.json"

# ---------- Helpers ----------
def extract_text_from_pdf(file_stream):
    """Extract text from PDF, fallback to OCR if page has no text."""
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    text = ""
    for page in doc:
        page_text = page.get_text().strip()
        if page_text:
            text += page_text + "\n"
        else:
            # OCR fallback for scanned PDFs
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text + "\n"
    return text.strip()

def analyze_lab_report(text):
    prompt = f"""
You are a medical lab report analyzer. Extract all tests, values, and reference ranges.
Flag each as Low/Normal/High, and generate overall health status and doctor's notes.
Return JSON strictly in this format:

{{
    "results": [
        {{
            "test": "Hemoglobin",
            "value": "13.5 g/dL",
            "range": "13-17 g/dL",
            "status": "Normal",
            "analysis": "Within normal range"
        }}
    ],
    "overall_status": "Healthy",
    "doctor_notes": "All parameters are within normal limits."
}}

Lab report text:
{text}
"""
    if not client.api_key:
        return {"error": "No API key provided"}
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    
    reply = response.choices[0].message.content
    try:
        return json.loads(reply)
    except:
        return {"error": "Failed to parse GPT JSON", "raw": reply}

def load_reports():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_reports(reports):
    with open(DATA_FILE, "w") as f:
        json.dump(reports, f, indent=4)

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    text = extract_text_from_pdf(file)
    analysis = analyze_lab_report(text)

    report_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": file.filename,
        "analysis": analysis
    }

    reports = load_reports()
    reports.append(report_record)
    save_reports(reports)

    if "results" in analysis:
        return jsonify(analysis)
    else:
        return jsonify({"error": "Analysis failed", "raw": analysis})

@app.route("/download/<int:index>")
def download(index):
    reports = load_reports()
    if index < 0 or index >= len(reports):
        return "Invalid report index", 400

    report = reports[index]
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Lab Report Analysis</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Date: {report['timestamp']}", styles["Normal"]))
    story.append(Spacer(1, 12))

    analysis = report['analysis']
    if "results" in analysis:
        for item in analysis["results"]:
            story.append(Paragraph(f"{item['test']}: {item['value']} ({item['range']}) - {item['status']}", styles["Normal"]))
            story.append(Paragraph(f"Analysis: {item['analysis']}", styles["Normal"]))
            story.append(Spacer(1, 6))
        story.append(Paragraph(f"Overall Status: {analysis.get('overall_status', '')}", styles["Normal"]))
        story.append(Paragraph(f"Doctor Notes: {analysis.get('doctor_notes', '')}", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="lab_report.pdf", mimetype="application/pdf")

if __name__ == "__main__":
    app.run(debug=True)
