import streamlit as st
from rank import HybridChatBot
from deep_translator import GoogleTranslator

# ---------------- LANGUAGE DICTIONARY ---------------- #
indian_languages = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Bengali": "bn",
    "Odia": "or",
    "Urdu": "ur",
    "Assamese": "as",
    "Konkani": "kok",
    "Manipuri (Meitei)": "mni",
    "Sanskrit": "sa",
    "Kashmiri": "ks",
    "Dogri": "doi",
    "Santali": "sat",
    "Maithili": "mai",
    "Bodo": "brx",
    "Nepali": "ne",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
}

def translate_text(text, target_lang, source_lang="auto"):
    return GoogleTranslator(source=source_lang, target=target_lang).translate(text)

# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(page_title="🌐 Satyukt vertual assistant", layout="centered")

st.title("Hi, there! How can i help you?")
st.write("select your preferred language.")

# Sidebar: language selection
user_lang_name = st.sidebar.selectbox("Select your language", list(indian_languages.keys()), index=0)
USER_LANGUAGE = indian_languages[user_lang_name]

# Initialize bot
if "bot" not in st.session_state:
    st.session_state.bot = HybridChatBot()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Type your message..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Step 1: Translate query to English
    query = prompt
    if USER_LANGUAGE != "en":
        query = translate_text(query, target_lang="en")

    # Step 2: Bot search
    results = st.session_state.bot.search(query, top_k=5, alpha=0.8)
    best = results[0]
    answer = best["answer"]

    # Step 3: Translate answer back
    if USER_LANGUAGE != "en":
        answer = translate_text(answer, target_lang=USER_LANGUAGE)

    # Show bot response
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
