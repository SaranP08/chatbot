import streamlit as st
from rank import HybridChatBot
from deep_translator import GoogleTranslator
from recommender import QuestionRecommender

st.set_page_config(page_title="Hybrid ChatBot", layout="wide")

# Languages
indian_languages = {
    "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Kannada": "kn",
    "Malayalam": "ml", "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa", "Bengali": "bn",
    "Odia": "or", "Urdu": "ur", "Assamese": "as", "French": "fr", "Spanish": "es",
    "German": "de", "Italian": "it"
}

@st.cache_data
def translate_text(text, target_lang, source_lang="auto"):
    """Translates text, returning original text if translation is not needed or fails."""
    if not text or source_lang == target_lang or (target_lang == "en" and source_lang == "auto"):
        return text
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception:
        return text

# Load models
@st.cache_resource
def load_bot():
    """Loads the HybridChatBot model."""
    return HybridChatBot()

@st.cache_resource
def load_recommender():
    """Loads the QuestionRecommender model."""
    return QuestionRecommender(
        faiss_index_path="data/faiss.index",
        questions_path="data/questions.npy"
    )

bot = load_bot()

# Core logic function (doesn't modify session state directly)
def get_bot_response(query_en):
    """
    Searches for an answer and new recommendations based on an English query.
    Returns the English answer and a list of new English recommendations.
    """
    query_en = str(query_en).strip()
    if not query_en:
        return "Please enter a question.", []

    results = bot.search(query_en, top_k=5, alpha=0.8)
    answer_en = results[0].get('answer') if results else "I'm sorry, I couldn't find an answer."
    new_recommendations = st.session_state.recommender.recommend(query_en)
    return answer_en, new_recommendations

# --- Session State Initialization ---
if 'recommender' not in st.session_state:
    st.session_state.recommender = load_recommender()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_language" not in st.session_state:
    st.session_state.user_language = "en"
if "recommendations" not in st.session_state:
    st.session_state.recommendations = st.session_state.recommender.get_initial_questions()

# Handler for recommended questions (button clicks)
if "query_for_next_run" in st.session_state:
    query_en = st.session_state.pop("query_for_next_run")
    
    # Add the user's clicked question to the chat history
    st.session_state.messages.append({
        "role": "user",
        "content": translate_text(query_en, st.session_state.user_language)
    })
    
    # Get and display the bot's response
    answer_en, new_recs = get_bot_response(query_en)
    st.session_state.messages.append({
        "role": "assistant",
        "content": translate_text(answer_en, st.session_state.user_language)
    })
    st.session_state.recommendations = new_recs


# --- Sidebar ---
st.sidebar.title("Settings")
selected_language_name = st.sidebar.selectbox(
    "Select your language", list(indian_languages.keys()), index=0
)
st.session_state.user_language = indian_languages[selected_language_name]

# --- Main UI ---
st.title("Sat2Farm AI Assistant")
st.markdown("Ask me anything about Sat2Farm, or select a recommended question below.")

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Recommendations Section ---
st.markdown("---")
rec_header = translate_text("Or select from these common questions:", st.session_state.user_language)
st.subheader(rec_header)

action_items = []
if st.session_state.recommender.history:
    action_items.append(("Back to previous questions", "go_back"))
for rec_en in st.session_state.recommendations:
    action_items.append((rec_en, rec_en))

num_columns = 3
cols = st.columns(num_columns)
for i, (text_en, action_id) in enumerate(action_items):
    col = cols[i % num_columns]
    display_text = translate_text(text_en, st.session_state.user_language)
    key = f"rec_button_{i}_{action_id}"

    if col.button(display_text, key=key, use_container_width=True):
        if action_id == "go_back":
            st.session_state.recommendations = st.session_state.recommender.go_back()
        else:
            st.session_state.query_for_next_run = action_id
        st.rerun()

# --- Chat Input with Send Symbol ---
# This widget provides the text box and send button at the bottom of the screen.
placeholder = translate_text("Type your question here...", st.session_state.user_language)
if prompt := st.chat_input(placeholder):
    # 1. Add user's original message to the chat display
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Translate the user's prompt to English for the backend model
    query_en = translate_text(
        prompt,
        target_lang="en",
        source_lang=st.session_state.user_language
    )

    # 3. Get the bot's response and new recommendations
    answer_en, new_recs = get_bot_response(query_en)

    # 4. Translate the bot's English answer to the user's language and display
    st.session_state.messages.append({
        "role": "assistant",
        "content": translate_text(answer_en, st.session_state.user_language)
    })

    # 5. Update the recommendations
    st.session_state.recommendations = new_recs

    # 6. Rerun the app to display the new messages and recommendations
    st.rerun()