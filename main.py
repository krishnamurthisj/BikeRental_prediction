import streamlit as st
import requests
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

# ── LOAD CSS ─────────────────────────────────────
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="NeoKrish EV Rentals",
    page_icon="🛵",
    layout="centered"
)

# ── SESSION STATE ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "input"
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── LOAD RAG PIPELINE ─────────────────────────────────────────
@st.cache_resource
def load_rag_pipeline():
    loader = PyPDFLoader("neokrish_clean.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(chunks, embeddings)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=80
    )
    return db, llm


# ── TABS ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Bike Demand Prediction", "🤖 Customer Assistant"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — ML PREDICTION
# ══════════════════════════════════════════════════════════════
with tab1:

    if st.session_state.page == "input":

        st.title("🚲 Bike Demand Prediction App")
        st.write("Enter feature values to predict bike demand.")

        st.sidebar.header("Input Features")

        date = st.sidebar.date_input("Date", datetime.date.today())
        holiday = st.sidebar.selectbox("Holiday", [0, 1])
        workingday = st.sidebar.selectbox("Working Day", [0, 1])
        weather = st.sidebar.selectbox(
            "Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"]
        )
        season = st.sidebar.selectbox("Season", [1, 2, 3, 4])
        hr = st.sidebar.number_input("Hour", 0, 23, 0)
        weekday = st.sidebar.number_input("Weekday", 0, 6, 0)
        temp = st.sidebar.number_input("Temperature", value=0.0)
        atemp = st.sidebar.number_input("Feels-like Temperature", value=0.0)
        hum = st.sidebar.number_input("Humidity", 0.0, 1.0, 0.5)
        windspeed = st.sidebar.number_input("Windspeed", value=0.0)

        data = {
            "holiday": int(holiday),
            "workingday": int(workingday),
            "weathersit_Clear": 1 if weather == "Clear" else 0,
            "weathersit_Mist": 1 if weather == "Mist" else 0,
            "weathersit_Light_Snow": 1 if weather == "Light Snow" else 0,
            "weathersit_Heavy_Rain": 1 if weather == "Heavy Rain" else 0,
            "season": int(season),
            "hr": int(hr),
            "weekday": int(weekday),
            "temp": float(temp),
            "atemp": float(atemp),
            "hum": float(hum),
            "windspeed": float(windspeed),
            "day": date.day,
            "month": date.month,
            "year": date.year,
        }

        input_df = pd.DataFrame(data, index=[0])

        with st.expander("🔍 Model Input"):
            st.dataframe(input_df)

        if st.button("🔮 Predict Bike Demand"):
            url = "http://127.0.0.1:8000/predict"
            try:
                response = requests.post(url, json=data)
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.predicted_value = result["predicted_bike_rentals"]
                    st.session_state.model_input = input_df
                    st.session_state.page = "result"
                    st.rerun()
                else:
                    st.error(response.text)
            except requests.exceptions.ConnectionError:
                st.error("❌ FastAPI server not running. Start it with: uvicorn api:app --reload")

    if st.session_state.page == "result":

        st.title("📈 Prediction Result")

        if "model_input" in st.session_state:
            with st.expander("🔍 Model Input Used"):
                st.dataframe(st.session_state.model_input)

        predicted_value = st.session_state.predicted_value
        st.success(f"🚴 Estimated Bike Rentals: **{predicted_value}**")

        if predicted_value > 300:
            st.info("📈 High demand expected! Consider surge pricing today.")
        elif predicted_value > 150:
            st.info("📊 Moderate demand. Normal pricing applies.")
        else:
            st.info("📉 Low demand today. Discounts may help attract customers.")

        st.subheader("📊 Prediction Distribution")
        kde_data = pd.Series(
            [predicted_value * (1 + i / 100) for i in range(-10, 11)]
        )
        fig, ax = plt.subplots()
        sns.kdeplot(kde_data, fill=True)
        ax.set_xlabel("Bike Rentals")
        ax.set_ylabel("Density")
        st.pyplot(fig)

        if st.button("🔙 Go Back"):
            st.session_state.page = "input"
            st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 2 — RAG CHATBOT
# ══════════════════════════════════════════════════════════════
with tab2:

    st.title("🤖 NeoKrish Customer Assistant")
    st.caption("Ask me about pricing, membership, penalties, booking, and bike availability!")

    with st.spinner("Loading assistant... please wait"):
        db, llm = load_rag_pipeline()

    if "predicted_value" in st.session_state:
        st.info(f"📊 Today's predicted bike demand: **{st.session_state.predicted_value} rentals**")

    st.divider()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if query := st.chat_input("Ask your question here..."):

        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # STEP 1 — Greeting check (no RAG needed)
                greetings = ["hi", "hello", "hey", "hii", "helo", "hai",
                             "hi!", "hello!", "hey!", "good morning",
                             "good afternoon", "good evening"]

                if query.lower().strip() in greetings:
                    response = "Hello! I am NeoKrish, your AI assistant. How can I help you today?"

                else:
                    # STEP 2 — Search PDF (k=2 gives 2 chunks for better accuracy)
                    retrieved = db.similarity_search(query, k=3)

                    if not retrieved:
                        response = "Sorry sir, I currently support only bike rental related queries. Please contact customer support for more help."

                    else:
                        # STEP 3 — Build context from PDF chunks
                        context = "\n".join([doc.page_content for doc in retrieved])

                        # STEP 4 — Strict prompt to Groq
                        prompt = f"""You are NeoKrish, an EV bike rental assistant in India.
Answer ONLY from the CONTEXT below. Do not use your own knowledge.
Use Indian Rupees only. Never use dollars.
Answer in ONE short sentence only.
If the answer is not found in CONTEXT, say: Sorry sir, please contact customer support.

CONTEXT:
{context}

Customer: {query}
NeoKrish:"""

                        response = llm.invoke(prompt).content

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar
    with st.sidebar:
        st.header("💡 Try asking:")
        st.markdown("""
        - What is the price of TVS iQube?
        - What is the penalty for late return?
        - Tell me about Gold membership
        - How do I book a bike online?
        - Which bike is best for students?
        - What if my bike breaks down?
        - Which season has highest demand?
        """)
        st.divider()
        st.markdown("📞 **Support:** 9876543210")
        st.markdown("🌐 **www.neokrish.in**")