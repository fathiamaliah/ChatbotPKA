"""# **Evaluasi**"""

import time
from transformers import pipeline
import PyPDF2
from google.colab import files
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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
            summarized_chunks.append(chunk)
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

def evaluate_chatbot(chatbot, summarized_chunks):
    print("\n=== Evaluasi Chatbot ===")

    question = "siapa dekan filkom ub?"
    expected_answer = "WAYAN FIRDAUS MAHMUDY"

    start_time = time.time()
    answer = chatbot(question, summarized_chunks)
    response_time = time.time() - start_time

    print(f"\nPertanyaan: {question}")
    print(f"Jawaban Chatbot: {answer}")
    print(f"Jawaban yang Diharapkan: {expected_answer}")
    print(f"Waktu Respons: {response_time:.2f} detik")

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([expected_answer, answer])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    score = similarity * 100
    print(f"Kesamaan Jawaban: {score:.2f}%")

    if score > 75:
        print("\nChatbot memberikan jawaban yang memuaskan!")
    else:
        print("\nJawaban Chatbot kurang memuaskan.")

def run_chatbot():
    uploaded = files.upload()

    pdf_path = list(uploaded.keys())[0]
    text = extract_text_from_pdf(pdf_path)

    chunks = split_text_into_chunks(text, max_words=500)

    summarizer = pipeline("summarization", model="google/pegasus-xsum")  # Model publik untuk summarization
    summarized_chunks = summarize_chunks(chunks, summarizer)

    chatbot = setup_chatbot()

    print("Chatbot sudah siap! Silakan ajukan pertanyaan tentang isi PDF.\n")
    while True:
        question = input("Anda: ")
        if question.lower() in ['exit', 'quit']:
            print("Keluar dari chatbot. Sampai jumpa!")
            break
        answer = chatbot(question, summarized_chunks)
        print(f"Chatbot: {answer}")

    evaluate_chatbot(chatbot, summarized_chunks)

run_chatbot()
