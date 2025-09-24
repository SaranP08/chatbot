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
# This is crucial for making the app feel persistent
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_language" not in st.session_state:
    st.session_state.user_language = "en"
if "recommendations" not in st.session_state:
    st.session_state.recommendations = st.session_state.recommender.get_initial_questions()
if "live_suggestions" not in st.session_state:
    st.session_state.live_suggestions = []
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = ""


# --- UI Sidebar for Language Selection ---
st.sidebar.title("Settings")
selected_language_name = st.sidebar.selectbox(
    "Select your language",
    options=list(indian_languages.keys()),
    index=0 # Default to English
)
st.session_state.user_language = indian_languages[selected_language_name]

# --- Core Logic Functions ---
def handle_query(query_en):
    """Processes a query, gets an answer, updates recommendations, and stores in chat history."""
    # 1. Get answer from the bot
    results = bot.search(query_en, top_k=5, alpha=0.8)
    answer_en = results[0]['answer'] if results else "I'm sorry, I couldn't find an answer to that."

    # 2. Translate for display
    query_display = translate_text(query_en, target_lang=st.session_state.user_language)
    answer_display = translate_text(answer_en, target_lang=st.session_state.user_language)

    # 3. Add to chat history
    st.session_state.messages.append({"role": "user", "content": query_display})
    st.session_state.messages.append({"role": "assistant", "content": answer_display})

    # 4. Get new recommendations
    st.session_state.recommendations = st.session_state.recommender.recommend(query_en)

    # 5. Clear any temporary states
    st.session_state.live_suggestions = []
    st.session_state.prompt_input = ""

def update_suggestions():
    """Called via callback when the user types in the chat input."""
    query_text = st.session_state.get("prompt_input", "")
    if len(query_text) > 3:  # Only search if the query is reasonably long
        # Use high alpha for pure semantic search on partial text
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
# Display live suggestions if they exist
if st.session_state.live_suggestions:
    st.markdown("---")
    st.write("Suggestions based on your typing:")
    s_cols = st.columns(len(st.session_state.live_suggestions) or 1)
    for i, suggestion in enumerate(st.session_state.live_suggestions):
        if s_cols[i].button(suggestion, key=f"live_sugg_{i}"):
            handle_query(suggestion)
            st.rerun()

# The main chat input box at the bottom of the screen
prompt = st.chat_input(
    translate_text("Ask your question here...", st.session_state.user_language),
    key="prompt_input",
    on_change=update_suggestions
)

# This block executes only when the user hits Enter
if prompt:
    query_en = translate_text(prompt, target_lang="en", source_lang=st.session_state.user_language)
    handle_query(query_en)
    st.rerun()

# --- Main Recommendations Section ---
st.markdown("---")
rec_header = translate_text("Recommended Questions", target_lang=st.session_state.user_language)
st.subheader(rec_header, anchor=False)

# Gather all action buttons (Back + Recommendations)
action_items = []
if st.session_state.recommender.history:
    action_items.append(("Back to previous questions", "go_back"))
for rec_en in st.session_state.recommendations:
    action_items.append((rec_en, rec_en))

# Display action buttons in a compact grid
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