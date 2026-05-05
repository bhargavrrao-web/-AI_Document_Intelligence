"""
AI Document Intelligence API
Main entry point for the FastAPI backend.
"""
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db, DocumentResult
from pipeline import extract_text_from_image_or_pdf, extract_entities, get_embedding, answer_question, classify_document, quantum_analyze
from pydantic import BaseModel
import faiss
import numpy as np
import mlflow
import time

app = FastAPI(title="AI Document Intelligence API")

# Setup MLflow
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("document_extraction")

# Initialize FAISS Index (In-memory for demo)
DIMENSION = 384 # all-MiniLM-L6-v2 uses 384 dimensions
faiss_index = faiss.IndexFlatL2(DIMENSION)
doc_ids_mapping = []

class CorrectionRequest(BaseModel):
    id: int
    document_type: str = None
    name: str = None
    date: str = None
    amount: str = None
    skills: str = None
    summary: str = None
    key_phrases: str = None

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    start_time = time.time()
    
    contents = await file.read()
    
    try:
        # OCR
        text = extract_text_from_image_or_pdf(contents, file.filename)
        # NER
        entities = extract_entities(text)
        
        # Classification (Deep Learning Concept: Prediction + Score)
        doc_type, dl_score = classify_document(text)
        
        # Vector Embedding
        embedding = get_embedding(text)
        
        # Quantum Analysis
        q_result = quantum_analyze(embedding)
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param("filename", file.filename)
            mlflow.log_param("document_type", doc_type)
            mlflow.log_param("quantum_analysis", q_result)
            mlflow.log_metric("processing_time", time.time() - start_time)
            mlflow.log_metric("dl_confidence_score", dl_score)
            mlflow.log_dict(entities, "extracted_entities.json")
            
        # Save to DB
        db_result = DocumentResult(
            filename=file.filename,
            document_type=doc_type,
            quantum_analysis=q_result,
            name=entities.get("name"),
            date=entities.get("date"),
            amount=entities.get("amount"),
            skills=entities.get("skills"),
            summary=entities.get("summary"),
            key_phrases=entities.get("key_phrases"),
            full_text=text
        )
        db.add(db_result)
        db.commit()
        db.refresh(db_result)
        
        # Add to FAISS Vector Store
        faiss_index.add(np.array([embedding], dtype=np.float32))
        doc_ids_mapping.append(db_result.id)
        
        return {
            "message": "Extracted successfully", 
            "data": db_result,
            "dl_concept": {
                "model": "distilbart-mnli",
                "prediction": doc_type,
                "confidence_score": dl_score
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
def get_results(db: Session = Depends(get_db)):
    results = db.query(DocumentResult).order_by(DocumentResult.created_at.desc()).all()
    return results

class AskRequest(BaseModel):
    document_id: int
    question: str

@app.post("/ask")
def ask_document(req: AskRequest, db: Session = Depends(get_db)):
    doc = db.query(DocumentResult).filter(DocumentResult.id == req.document_id).first()
    if not doc or not doc.full_text:
        raise HTTPException(status_code=404, detail="Document or text not found")
    answer = answer_question(doc.full_text, req.question)
    return {"answer": answer}

@app.get("/search")
def search_documents(q: str, db: Session = Depends(get_db)):
    if not q or faiss_index.ntotal == 0:
        return []
    
    query_emb = get_embedding(q)
    k = min(5, faiss_index.ntotal)
    distances, indices = faiss_index.search(np.array([query_emb], dtype=np.float32), k)
    
    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(doc_ids_mapping):
            doc_id = doc_ids_mapping[idx]
            doc = db.query(DocumentResult).filter(DocumentResult.id == doc_id).first()
            if doc:
                results.append(doc)
    return results

@app.put("/correct")
def update_correction(request: CorrectionRequest, db: Session = Depends(get_db)):
    db_result = db.query(DocumentResult).filter(DocumentResult.id == request.id).first()
    if not db_result:
        raise HTTPException(status_code=404, detail="Document not found")
        
    if request.name is not None: db_result.name = request.name
    if request.date is not None: db_result.date = request.date
    if request.amount is not None: db_result.amount = request.amount
    if request.skills is not None: db_result.skills = request.skills
    if request.summary is not None: db_result.summary = request.summary
    if request.key_phrases is not None: db_result.key_phrases = request.key_phrases
    
    db.commit()
    db.refresh(db_result)
    
    with mlflow.start_run():
        mlflow.log_param("correction_id", db_result.id)
        mlflow.log_param("corrected_name", request.name)
        
    return {"message": "Updated successfully", "data": db_result}
