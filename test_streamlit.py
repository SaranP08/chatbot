import streamlit as st
from rank import HybridChatBot
from deep_translator import GoogleTranslator
from recommender import QuestionRecommender
from streamlit_searchbox import st_searchbox

# --- App Configuration ---
st.set_page_config(page_title="Hybrid ChatBot", layout="wide")

# --- Language & Translation Setup ---
# (This section remains unchanged)
indian_languages = {
    "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Kannada": "kn", 
    "Malayalam": "ml", "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa", "Bengali": "bn", 
    "Odia": "or", "Urdu": "ur", "Assamese": "as", "French": "fr", "Spanish": "es", 
    "German": "de", "Italian": "it"
}

@st.cache_data
def translate_text(text, target_lang, source_lang="auto"):
    if not text or source_lang == target_lang or (target_lang == "en" and source_lang == "auto"):
        return text
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        st.error(f"Translation Error: {e}")
        return text

# --- Model and Recommender Loading ---
# (This section remains unchanged)
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
# (This section remains unchanged)
st.sidebar.title("Settings")
selected_language_name = st.sidebar.selectbox(
    "Select your language",
    options=list(indian_languages.keys()),
    index=0 
)
st.session_state.user_language = indian_languages[selected_language_name]

# --- Core Logic Functions ---
def handle_query(query_en):
    query_en = str(query_en)
    results = bot.search(query_en, top_k=5, alpha=0.8)
    answer_en = results[0].get('answer') if results else "I'm sorry, I couldn't find an answer to that."

    query_display = translate_text(query_en, target_lang=st.session_state.user_language)
    answer_display = translate_text(answer_en, target_lang=st.session_state.user_language)

    st.session_state.messages.append({"role": "user", "content": query_display})
    st.session_state.messages.append({"role": "assistant", "content": answer_display})

    st.session_state.recommendations = st.session_state.recommender.recommend(query_en)

def get_suggestions(query, **kwargs):
    if len(query) < 3:
        return []
    all_questions = st.session_state.recommender.questions
    suggestions = [q for q in all_questions if query.lower() in q.lower()]
    return suggestions[:5]

# --- Main App Interface ---
st.title("Sat2Farm AI Assistant", anchor=False)
st.markdown("Ask me anything about Sat2Farm, or select a recommended question below.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Your working initialization logic is here ---
if "question_searchbox" not in st.session_state or st.session_state.question_searchbox is None:
    st.session_state.question_searchbox = {
        "options_js": [], "value": "", "key_react": "init", "result": None, "search": None
    }

# --- User Input Section with Dual Functionality ---
# Use columns to place the searchbox and submit button side-by-side
input_col, button_col = st.columns([4, 1])

with input_col:
    selected_suggestion = st_searchbox(
        search_function=get_suggestions,
        placeholder="Type here for suggestions or to ask a question...",
        key="question_searchbox"
    )

# Logic Path 1: User clicks the "Submit" button
with button_col:
    if st.button("Submit", use_container_width=True, type="primary"):
        # Read the raw text that the user typed from the component's state
        raw_query = st.session_state.question_searchbox.get("search")
        if raw_query:
            handle_query(raw_query)
            st.session_state.question_searchbox = None  # Clear state
            st.rerun()

# Logic Path 2: User clicks on a suggestion from the dropdown list
if selected_suggestion:
    handle_query(selected_suggestion)
    st.session_state.question_searchbox = None  # Clear state
    st.rerun()


# --- Main Recommendations Section ---
st.markdown("---")
rec_header = translate_text("Or select from these common questions:", target_lang=st.session_state.user_language)
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