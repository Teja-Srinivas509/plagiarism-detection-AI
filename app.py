import os
import io
import re
import docx
import nltk
import requests
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from checker import BERTPlagiarismChecker
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# NLP Setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Plagiarism Checker
plagiarism_checker = BERTPlagiarismChecker()

# Google Custom Search API
GOOGLE_API_KEY = "AIzaSyAQu1cit5tMW5oX1ZvQXy947MwCQt2U0Nw"
SEARCH_ENGINE_ID = "103b19c7f153f4eb8"

# ------------------ Helper Functions ------------------

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])


def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = nltk.word_tokenize(text)
    filtered_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(filtered_words)


def search_google(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"
    response = requests.get(url)
    data = response.json()
    results = []
    if "items" in data:
        for item in data["items"]:
            results.append({"title": item["title"], "link": item["link"]})
    return results

# ------------------ Routes ------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/main")
def main_page():
    return render_template("main.html")


@app.route("/compare", methods=["POST"])
def compare_files():
    file1 = request.files.get("file1")
    file2 = request.files.get("file2")

    if not file1 or not file2:
        return "Both files are required.", 400

    def extract(file):
        if file.filename.endswith('.pdf'):
            return extract_text_from_pdf(file)
        elif file.filename.endswith('.docx'):
            return extract_text_from_docx(file)
        else:
            return file.read().decode("utf-8")

    text1 = extract(file1)
    text2 = extract(file2)

    pre1 = preprocess(text1)
    pre2 = preprocess(text2)

    final_score, status = plagiarism_checker.check_similarity(pre1, pre2)

    similarity_text = f"{final_score:.2f}%"

    return render_template("compare.html", 
        similarity=similarity_text,
        message=status,
        uploaded_text1=pre1,
        uploaded_text2=pre2
    )


@app.route("/send", methods=["POST"])
def check_text():
    text1 = request.form.get("text1", "").strip()
    text2 = request.form.get("text2", "").strip()

    if not text1 or not text2:
        return jsonify({"error": "Enter text in both fields"}), 400

    pre1 = preprocess(text1)
    pre2 = preprocess(text2)

    final_score, status = plagiarism_checker.check_similarity(pre1, pre2)
    similarity_text = f"{final_score:.2f}%"

    return jsonify({
        "similarity": float(final_score),
        "similarity_text": similarity_text,
        "message": status
    })


@app.route("/check_plagiarism", methods=["POST"])
def check_online():
    text = None

    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        filename = file.filename.lower()
        try:
            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(io.BytesIO(file.read()))
            elif filename.endswith(".docx"):
                text = extract_text_from_docx(file)
            elif filename.endswith(".txt"):
                text = file.read().decode("utf-8")
            else:
                return jsonify({"error": "Unsupported file format"}), 400
        except Exception as e:
            return jsonify({"error": f"Could not extract text: {str(e)}"}), 500

    elif "text" in request.form:
        text = request.form["text"].strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    text = preprocess(text)
    urls = search_google(text[:300])

    if not urls:
        return jsonify({"message": "No matching URLs found. Content appears original."})
    return jsonify({"matched_urls": urls})


if __name__ == "__main__":
    app.run(debug=True)
