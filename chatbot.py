# -*- coding: utf-8 -*-


Original file is located at
https://colab.research.google.com/drive/1jhM5MVPbu80yaRDqttDhlemopzeT-WpM?usp=sharing"""

!pip install transformers datasets pypdf2 sentence-transformers
import PyPDF2
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

!pip install streamlit
!pip install pyngrok


%%writefile app.py
import streamlit as st
import PyPDF2
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text_into_chunks(text, max_words=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def summarize_chunks(chunks, summarizer_pipeline):
    summarized_chunks = []
    for chunk in chunks:
        try:
            summarized = summarizer_pipeline(chunk, max_length=512, min_length=30, do_sample=False)
            summarized_chunks.append(summarized[0]['summary_text'])
        except Exception as e:
            summarized_chunks.append(chunk)  # If summarization fails, use original text
    return summarized_chunks

def setup_chatbot():
    qa_pipeline = pipeline("question-answering", model="deepset/xlm-roberta-large-squad2")

    def chatbot(question, chunks):
        best_answer = ""
        best_score = 0
        for chunk in chunks:
            try:
                result = qa_pipeline(question=question, context=chunk)
                if result['score'] > best_score:
                    best_score = result['score']
                    best_answer = result['answer']
            except Exception as e:
                continue
        return best_answer

    return chatbot

def run_gui():
    st.title("Chatbot Layanan Administrasi PerkuliahanBerbasis Multi-Agen AI untuk Pengelolaan Informasi Akademik")
    st.write("Unggah PDF Buku Pedoman Perkuliahan FILKOM dan tanyakan pertanyaan yang ingin ditanyakan")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)

        chunks = split_text_into_chunks(text, max_words=500)

        with st.spinner("Summarizing PDF content..."):
            summarizer = pipeline("summarization", model="google/pegasus-xsum")
            summarized_chunks = summarize_chunks(chunks, summarizer)

        chatbot = setup_chatbot()
        st.success("Chatbot siap digunakan!")

        question = st.text_input("Pertanyaan Anda:")
        if question:
            with st.spinner("Mencari jawaban..."):
                answer = chatbot(question, summarized_chunks)
            st.write("Jawaban")
            st.write(answer)

if __name__ == "__main__":
    run_gui()


"""# Streamlit Set"""

from pyngrok import ngrok

ngrok.set_auth_token("2qWofkl8bXBvICrtwtCjxOmSwAx_3VKQJ19APXjoomjrsMCBK")

import os
public_url = ngrok.connect(8501)
os.system("streamlit run app.py &")
public_url
