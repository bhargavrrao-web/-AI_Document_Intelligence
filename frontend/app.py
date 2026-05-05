import streamlit as st
import requests
import pandas as pd
import os

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Intelligence", page_icon="📄", layout="wide")

# Load Custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("style.css")
except FileNotFoundError:
    pass

st.title("✨ AI Document Intelligence")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 2rem;'>Extract, Summarize, and Analyze your documents with Deep Learning.</p>", unsafe_allow_html=True)

# Tabs for Upload and History
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Extract", "🗄️ Database History", "🔍 Semantic Search", "⚛️ Quantum Intelligence"])

with tab1:
    st.markdown("### 1. Upload Document")
    uploaded_file = st.file_uploader("Upload an Image or PDF", type=["png", "jpg", "jpeg", "pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.button("Analyze Document ✨", use_container_width=True):
            with st.spinner("Analyzing document... This might take a moment as our AI models process the text."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    try:
                        response = requests.post(f"{BACKEND_URL}/upload", files=files)
                        if response.status_code == 200:
                            data = response.json()["data"]
                            st.session_state['current_extraction'] = data
                            st.success("Extraction Complete!")
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Failed to connect to backend: {e}")

    if 'current_extraction' in st.session_state:
        st.markdown("---")
        st.markdown("### 2. Analysis Results")
        data = st.session_state['current_extraction']
        
        # Top section for Summary and Key Phrases
        st.info(f"**📑 Document Type:** {data.get('document_type', 'Unknown')}")
        st.warning(f"**⚛️ Quantum Security Analysis:** {data.get('quantum_analysis', 'No quantum data available.')}")
        col_sum, col_key = st.columns([2, 1])
        with col_sum:
            st.info("**📝 Document Summary**")
            st.write(data.get("summary") or "No summary generated.")
        with col_key:
            st.success("**🔑 Important Topics**")
            st.write(data.get("key_phrases") or "No key phrases extracted.")
            
        st.markdown("### 3. Review & Edit Fields")
        with st.form("correction_form"):
            col_form1, col_form2 = st.columns(2)
            with col_form1:
                name = st.text_input("Name (PERSON)", value=data.get("name") or "")
                date = st.text_input("Date (DATE)", value=data.get("date") or "")
            with col_form2:
                amount = st.text_input("Amount (MONEY)", value=data.get("amount") or "")
                skills = st.text_area("Skills", value=data.get("skills") or "")
            
            # Hidden fields for new data
            summary = data.get("summary")
            key_phrases = data.get("key_phrases")
            
            submitted = st.form_submit_button("Save Analysis 💾", use_container_width=True)
            
            if submitted:
                update_data = {
                    "id": data["id"],
                    "name": name,
                    "date": date,
                    "amount": amount,
                    "skills": skills,
                    "summary": summary,
                    "key_phrases": key_phrases
                }
                try:
                    res = requests.put(f"{BACKEND_URL}/correct", json=update_data)
                    if res.status_code == 200:
                        st.success("Analysis saved successfully!")
                        st.session_state['current_extraction'] = res.json()["data"]
                    else:
                        st.error("Failed to save corrections.")
                except Exception as e:
                    st.error(f"Connection error: {e}")
                    
        st.markdown("### 4. 💬 Chat with Document")
        question = st.text_input("Ask a question about this document:", key="chat_input")
        if st.button("Ask AI ✨", use_container_width=True, key="chat_btn"):
            if question:
                with st.spinner("Thinking..."):
                    try:
                        res = requests.post(f"{BACKEND_URL}/ask", json={"document_id": data["id"], "question": question})
                        if res.status_code == 200:
                            st.success(f"**Answer:** {res.json()['answer']}")
                        else:
                            st.error("Failed to get an answer.")
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.markdown("<div style='padding: 3rem; text-align: center; border: 1px dashed rgba(129, 140, 248, 0.5); border-radius: 12px; color: #94a3b8; margin-top: 2rem;'>Upload a document to see the AI Analysis here.</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("### Extraction Database")
    if st.button("Refresh Data 🔄"):
        pass # Streamlit reruns on button click anyway
        
    try:
        res = requests.get(f"{BACKEND_URL}/results")
        if res.status_code == 200:
            results = res.json()
            if results:
                df = pd.DataFrame(results)
                # Reorder and format columns
                df = df[['id', 'filename', 'document_type', 'quantum_analysis', 'name', 'date', 'amount', 'skills', 'key_phrases', 'summary', 'created_at']]
                df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Show some metrics
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Documents Analyzed", len(df))
                col2.metric("Names Extracted", df['name'].notna().sum())
                col3.metric("Summaries Generated", df['summary'].notna().sum())
                
            else:
                st.info("No data available yet.")
        else:
            st.error("Failed to load data.")
    except Exception as e:
        st.error(f"Connection error: {e}")

with tab3:
    st.markdown("### Semantic Document Search")
    st.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>Search using concepts instead of exact keywords! Powered by FAISS Vector Database.</p>", unsafe_allow_html=True)
    
    search_query = st.text_input("Search for concepts, skills, or topics:")
    if st.button("Search 🔍", use_container_width=True, key="search_btn"):
        if search_query:
            with st.spinner("Searching vector database..."):
                try:
                    res = requests.get(f"{BACKEND_URL}/search", params={"q": search_query})
                    if res.status_code == 200:
                        search_results = res.json()
                        if search_results:
                            for idx, r in enumerate(search_results):
                                with st.expander(f"📄 {r['filename']} (Result {idx+1})"):
                                    st.write(f"**Name:** {r.get('name') or 'N/A'}")
                                    st.write(f"**Skills:** {r.get('skills') or 'N/A'}")
                                    st.write(f"**Summary:** {r.get('summary') or 'N/A'}")
                                    st.write(f"**Key Phrases:** {r.get('key_phrases') or 'N/A'}")
                        else:
                            st.info("No matching documents found.")
                    else:
                        st.error("Search failed.")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab4:
    st.markdown("### ⚛️ Quantum Machine Learning Insights")
    st.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>Explore the quantum state analysis of your documents using Variational Quantum Circuits (VQC) and Data Re-uploading.</p>", unsafe_allow_html=True)
    
    col_q1, col_q2 = st.columns([1, 1])
    
    with col_q1:
        st.image("quantum_viz.png", use_container_width=True)
        st.markdown("""
        **Quantum Circuit Concepts:**
        - **Data Re-uploading**: Non-linear feature mapping via multi-layer entanglement.
        - **VQC (Variational Quantum Circuit)**: Optimization of quantum parameters for anomaly detection.
        - **Expectation Values**: Measuring Pauli-Z operators across 6 qubits for document fingerprinting.
        """)
        
    with col_q2:
        st.info("**Latest Quantum Signatures**")
        try:
            res = requests.get(f"{BACKEND_URL}/results")
            if res.status_code == 200:
                results = res.json()
                if results:
                    latest = results[0]
                    st.metric("Current Document", latest['filename'])
                    st.success(f"**Status:** {latest['quantum_analysis']}")
                    
                    # Simulated Quantum Metrics for Visualization
                    st.markdown("#### Quantum Metrics")
                    st.progress(0.85, text="Quantum Entanglement Depth")
                    st.progress(0.72, text="Circuit Fidelity")
                    st.progress(0.94, text="Gate Reliability")
                else:
                    st.write("No documents analyzed yet.")
        except:
            st.error("Failed to load quantum data.")
            
    st.markdown("---")
    st.markdown("### 🛠️ Advanced QML Configuration")
    st.slider("Number of Qubits", 2, 10, 6, disabled=True)
    st.slider("Variational Layers", 1, 5, 3, disabled=True)
    st.checkbox("Enable Hybrid Quantum-Classical Feedback Loop", value=True, disabled=True)
