import pytesseract
from pdf2image import convert_from_bytes
import spacy
from PIL import Image
import io
import re
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer
from quantum_logic import q_classifier

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model for the spacy")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

# Load Generative LLM for Chat
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

# Load Zero-Shot Classifier
classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3", device=-1)

# Load Embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_image_or_pdf(file_bytes, filename):
    text = ""
    if filename.lower().endswith('.pdf'):
        images = convert_from_bytes(file_bytes)
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
    else:
        # Assume it's an image
        img = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img)
    return text

def extract_entities(text):
    doc = nlp(text)
    
    # Simple extraction logic based on NER
    name = None
    date = None
    amount = None
    skills = []
    
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not name:
            name = ent.text
        elif ent.label_ == "DATE" and not date:
            date = ent.text
        elif ent.label_ == "MONEY" and not amount:
            amount = ent.text
            
    # Regex fallback for amounts if spacy missed it
    if not amount:
        amount_match = re.search(r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?', text)
        if amount_match:
            amount = amount_match.group(0)

    # Simple keyword-based skill extraction for resumes
    common_skills = ['python', 'java', 'c++', 'sql', 'machine learning', 'data science', 'fastapi', 'docker', 'aws', 'azure', 'react', 'javascript']
    text_lower = text.lower()
    for skill in common_skills:
        if skill in text_lower:
            skills.append(skill.capitalize())
            
    # Extract Key Phrases (Noun Chunks) using spaCy
    noun_chunks = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1 and len(chunk.text) > 4]
    key_phrases = list(set(noun_chunks))[:10]  # Take top 10 unique key phrases
    
    # Generate Summary using transformers
    summary = None
    if len(text.split()) > 30:
        # Limit text length to prevent OOM
        try:
            summary_out = summarizer(text[:1024], max_length=130, min_length=30, do_sample=False)
            summary = summary_out[0]['summary_text']
        except Exception as e:
            print(f"Summarization error: {e}")
            summary = "Summarization failed or text too short."
    else:
        summary = "Text is too short for summarization."

    return {
        "name": name,
        "date": date,
        "amount": amount,
        "skills": ", ".join(skills) if skills else None,
        "key_phrases": ", ".join(key_phrases) if key_phrases else None,
        "summary": summary
    }

def get_embedding(text):
    if not text:
        return [0.0] * 384
    return embedder.encode(text).tolist()

def answer_question(context, question):
    if not context or len(context) < 10:
        return "Context is too short to answer questions."
    try:
        prompt = f"Based on the following document, answer the question accurately.\n\nDocument:\n{context[:2500]}\n\nQuestion: {question}\nAnswer:"
        result = qa_pipeline(prompt, max_length=150, do_sample=False)
        return result[0]['generated_text']
    except Exception as e:
        print(f"QA Error: {e}")
        return "Sorry, I couldn't process the question."

def classify_document(text):
    if not text or len(text) < 10:
        return "Unknown"
    labels = ["Invoice", "Resume", "Receipt", "Contract", "Article", "Report", "Form"]
    try:
        result = classifier(text[:1024], candidate_labels=labels)
        return result['labels'][0]
    except Exception as e:
        print(f"Classification error: {e}")
        return "Unknown"

def quantum_analyze(embedding):
    try:
        label, score = q_classifier.analyze_document(embedding)
        return f"{label} (Q-Value: {score:.4f})"
    except Exception as e:
        print(f"Quantum Error: {e}")
        return "Quantum analysis failed."
