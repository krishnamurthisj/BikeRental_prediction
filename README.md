# BikeRental_prediction
# 🛵 NeoKrish EV Rentals — AI-Powered Bike Rental Platform

> An end-to-end intelligent bike rental platform combining Machine Learning, RAG-based AI Chatbot, and NLP Sentiment Analysis — built with Python, Streamlit, LangChain, and Groq LLaMA3.

---

## 🚀 Live Demo

![NeoKrish EV Rentals](https://img.shields.io/badge/Status-Live-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-purple)

---

## 📌 Project Overview

NeoKrish EV Rentals is a smart bike rental platform that uses AI and Machine Learning to help both customers and business owners. The platform has three main features:

- **Bike Demand Prediction** — predicts how many bikes will be rented based on weather, season, time, and other factors
- **AI Customer Assistant** — a RAG-based chatbot that answers customer questions about pricing, membership, penalties, and booking using a business knowledge PDF
- **Customer Sentiment Analysis** — analyzes customer chat conversations to identify if the customer is happy or unhappy using VADER, AFINN, and TextBlob

---

## 🎯 Features

### 📊 Tab 1 — Bike Demand Prediction
- Predicts total bike rentals using a trained Machine Learning model
- Input features: date, season, weather, temperature, humidity, windspeed, hour
- Displays prediction result with demand level and business recommendation
- Shows KDE distribution chart of predicted demand
- Backend powered by FastAPI

### 🤖 Tab 2 — AI Customer Assistant (RAG Chatbot)
- Answers customer questions based on business policy PDF
- Uses LangChain + FAISS vector store for document retrieval
- Powered by Groq LLaMA3 for fast and accurate responses
- Handles greetings separately without RAG search
- Strict PDF-only answers — no external knowledge used
- Response time under 2 seconds using Groq cloud API

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Frontend | Streamlit |
| Backend API | FastAPI + Uvicorn |
| Machine Learning | Scikit-learn, LightGBM, Pandas, NumPy |
| RAG Pipeline | LangChain, FAISS, HuggingFace Embeddings |
| LLM | Groq LLaMA3 (llama-3.1-8b-instant) |
| PDF Processing | PyPDF, ReportLab |
| Visualization | Matplotlib, Seaborn, Plotly |
| Environment | Python-dotenv |

---

## 📁 Project Structure

```
NeoKrish-EV-Rental/
├── main.py                  # Main Streamlit application (3 tabs)
├── api.py                   # FastAPI backend for ML prediction
├── neokrish_clean.pdf       # Business policy knowledge base for RAG
├── requirements.txt         # All dependencies
├── .env                     # Environment variables (not pushed to GitHub)
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

---

## ⚙️ Installation and Setup

### Step 1 — Clone the repository
```bash
git clone https://github.com/krishnamurthisj/NeoKrish-EV-Rental.git
cd NeoKrish-EV-Rental
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Set up environment variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free Groq API key at: https://console.groq.com

### Step 4 — Run FastAPI backend
```bash
uvicorn api:app --reload
```

### Step 5 — Run Streamlit app
```bash
streamlit run main.py
```

### Step 6 — Open browser
```
http://localhost:8501
```

---

## 📦 Requirements

```
streamlit
fastapi
uvicorn
langchain
langchain-community
langchain-groq
faiss-cpu
sentence-transformers
pypdf
pandas
numpy
scikit-learn
lightgbm
matplotlib
seaborn
plotly
reportlab
python-dotenv
```

---

## 🔄 How RAG Pipeline Works

```
Customer Question
      ↓
FAISS searches neokrish_clean.pdf chunks
      ↓
Top 3 relevant chunks retrieved
      ↓
Chunks sent as context to Groq LLaMA3
      ↓
LLaMA3 answers based ONLY on PDF context
      ↓
Short accurate answer shown to customer
```

---


```

---

## 🖥️ Screenshots

> Add screenshots of your app here after running it locally.
> Tab 1 — Prediction, Tab 2 — Chatbot

---

## 👨‍💻 Author

**Krishna Murthi S J**
- 📧 Krishnamurthisj3@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/krishna-murthi-s-j)
- 🐙 [GitHub](https://github.com/krishnamurthisj)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [LangChain](https://langchain.com) — RAG pipeline framework
- [Groq](https://groq.com) — Fast LLaMA3 inference API
- [HuggingFace](https://huggingface.co) — Sentence embeddings
- [Streamlit](https://streamlit.io) — Web app framework

---

⭐ **If you found this project useful, please give it a star on GitHub!**
