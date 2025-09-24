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

# --- UI Sidebar for Language Selection ---
st.sidebar.title("Settings")
selected_language_name = st.sidebar.selectbox(
    "Select your language",
    options=list(indian_languages.keys()),
    index=0 # Default to English
)
st.session_state.user_language = indian_languages[selected_language_name]

# --- Core Logic Function ---
def handle_query(query_en):
    """Processes a query, gets an answer, updates recommendations, and stores in chat history."""
    # 1. Get answer from the bot
    results = bot.search(query_en, top_k=5, alpha=0.8)
    if not results:
        answer_en = "I'm sorry, I couldn't find an answer to that."
    else:
        answer_en = results[0]['answer']

    # 2. Translate for display
    query_display = translate_text(query_en, target_lang=st.session_state.user_language)
    answer_display = translate_text(answer_en, target_lang=st.session_state.user_language)

    # 3. Add to chat history
    st.session_state.messages.append({"role": "user", "content": query_display})
    st.session_state.messages.append({"role": "assistant", "content": answer_display})

    # 4. Get new recommendations and update state
    st.session_state.recommendations = st.session_state.recommender.recommend(query_en)

# --- Main App Interface ---
st.title("Sat2Farm AI Assistant", anchor=False)
st.markdown("Ask me anything about Sat2Farm, or select a recommended question below.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Display Recommendations and "Back" button ---
# --- Display Recommendations and "Back" button ---
st.markdown("---")
rec_header = translate_text("Recommended Questions", target_lang=st.session_state.user_language)
st.subheader(rec_header, anchor=False) # anchor=False removes the link icon

# We will handle the back button and recommendations in a grid layout
action_items = []
# Add the "Back" action first if history exists
if st.session_state.recommender.history:
    action_items.append(("Back to previous questions", "go_back"))

# Add the recommended questions
for rec_en in st.session_state.recommendations:
    action_items.append((rec_en, rec_en)) # Use the question itself as the ID

# You can adjust this number to change how many buttons appear per row
num_columns = 3 
cols = st.columns(num_columns)

# Distribute the action items across the columns
for i, (text_en, action_id) in enumerate(action_items):
    col = cols[i % num_columns]
    display_text = translate_text(text_en, target_lang=st.session_state.user_language)
    
    if col.button(display_text, key=action_id, use_container_width=True):
        if action_id == "go_back":
            st.session_state.recommendations = st.session_state.recommender.go_back()
        else:
            # The action_id is the English version of the question
            handle_query(action_id)
        st.rerun()