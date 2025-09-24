import streamlit as st
from rank import HybridChatBot
from deep_translator import GoogleTranslator
from recommender import QuestionRecommender

# --- App Configuration ---
st.set_page_config(page_title="Hybrid ChatBot", layout="wide")

# --- Language & Translation Setup ---
indian_languages = {
    "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Kannada": "kn", 
    "Malayalam": "ml", "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa", "Bengali": "bn", 
    "Odia": "or", "Urdu": "ur", "Assamese": "as", "French": "fr", "Spanish": "es", 
    "German": "de", "Italian": "it"
}

@st.cache_data
def translate_text(text, target_lang, source_lang="auto"):
    if source_lang == target_lang or (target_lang == "en" and source_lang == "auto"):
        return text
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        st.error(f"Translation Error: {e}")
        return text

# --- Model and Recommender Loading ---
@st.cache_resource
def load_bot():
    return HybridChatBot()

@st.cache_resource
def load_recommender():
    return QuestionRecommender(
        faiss_index_path="data/faiss.index",
        questions_path="data/questions.npy"
    )

bot = load_bot()
if 'recommender' not in st.session_state:
    st.session_state.recommender = load_recommender()

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_language" not in st.session_state:
    st.session_state.user_language = "en"
if "recommendations" not in st.session_state:
    st.session_state.recommendations = st.session_state.recommender.get_initial_questions()
if "live_suggestions" not in st.session_state:
    st.session_state.live_suggestions = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# --- UI Sidebar for Language Selection ---
st.sidebar.title("Settings")
selected_language_name = st.sidebar.selectbox(
    "Select your language",
    options=list(indian_languages.keys()),
    index=0 
)
st.session_state.user_language = indian_languages[selected_language_name]

# --- Core Logic Functions ---
def handle_query(query_en):
    results = bot.search(query_en, top_k=5, alpha=0.8)
    answer_en = results[0]['answer'] if results else "I'm sorry, I couldn't find an answer to that."

    query_display = translate_text(query_en, target_lang=st.session_state.user_language)
    answer_display = translate_text(answer_en, target_lang=st.session_state.user_language)

    st.session_state.messages.append({"role": "user", "content": query_display})
    st.session_state.messages.append({"role": "assistant", "content": answer_display})

    st.session_state.recommendations = st.session_state.recommender.recommend(query_en)
    
    st.session_state.live_suggestions = []
    st.session_state.user_input = ""

def update_suggestions():
    query_text = st.session_state.get("user_input", "")
    if len(query_text) > 3:
        results = bot.search(query_text, top_k=3, alpha=1.0)
        st.session_state.live_suggestions = [res['question'] for res in results]
    else:
        st.session_state.live_suggestions = []

# --- Main App Interface ---
st.title("Sat2Farm AI Assistant", anchor=False)
st.markdown("Ask me anything about Sat2Farm, or select a recommended question below.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Live Suggestions Section ---

# THE FIX IS HERE: We remove the st.form and use st.columns for layout.
input_col, button_col = st.columns([4, 1])

with input_col:
    st.text_input(
        "Ask your question here...",
        key="user_input",
        on_change=update_suggestions,
        autocomplete="off",
        label_visibility="collapsed" # Hides the label to make it cleaner
    )

with button_col:
    if st.button("Submit", use_container_width=True, type="primary"):
        prompt = st.session_state.user_input
        if prompt:
            query_en = translate_text(prompt, target_lang="en", source_lang=st.session_state.user_language)
            handle_query(query_en)
            st.rerun()

# Display live suggestions if they exist
if st.session_state.live_suggestions:
    st.markdown("---")
    st.write("Suggestions:")
    for suggestion in st.session_state.live_suggestions:
        if st.button(suggestion, key=f"live_sugg_{suggestion}", use_container_width=True):
            handle_query(suggestion)
            st.rerun()

# --- Main Recommendations Section ---
st.markdown("---")
rec_header = translate_text("Recommended Questions", target_lang=st.session_state.user_language)
st.subheader(rec_header, anchor=False)

action_items = []
if st.session_state.recommender.history:
    action_items.append(("Back to previous questions", "go_back"))
for rec_en in st.session_state.recommendations:
    action_items.append((rec_en, rec_en))

num_columns = 3 
cols = st.columns(num_columns)
for i, (text_en, action_id) in enumerate(action_items):
    col = cols[i % num_columns]
    display_text = translate_text(text_en, target_lang=st.session_state.user_language)
    
    if col.button(display_text, key=action_id, use_container_width=True):
        if action_id == "go_back":
            st.session_state.recommendations = st.session_state.recommender.go_back()
        else:
            handle_query(action_id)
        st.rerun()