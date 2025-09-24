from rank import HybridChatBot
from deep_translator import GoogleTranslator
# Make sure to import the updated recommender
from recommender import QuestionRecommender 

# --- Language setup (remains the same) ---
indian_languages = {
    "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Kannada": "kn", "Malayalam": "ml",
    "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa", "Bengali": "bn", "Odia": "or",
    "Urdu": "ur", "Assamese": "as", "Konkani": "kok", "Manipuri (Meitei)": "mni",
    "Sanskrit": "sa", "Kashmiri": "ks", "Dogri": "doi", "Santali": "sat",
    "Maithili": "mai", "Bodo": "brx", "Nepali": "ne", "English": "en",
    "French": "fr", "Spanish": "es", "German": "de", "Italian": "it"
}

def translate_text(text, target_lang, source_lang="auto"):
    if source_lang == target_lang or target_lang == "en" and source_lang == "auto":
        return text
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

if __name__ == "__main__":
    bot = HybridChatBot()
    recommender = QuestionRecommender(
        faiss_index_path="data/faiss.index",
        questions_path="data/questions.npy"
    )

    print("Available languages:", ", ".join(indian_languages.keys()))
    language_choice = input("Select your language (e.g., Hindi, English, Tamil): ").strip().title()
    USER_LANGUAGE = indian_languages.get(language_choice, "en")
    print(f"Language set to: {language_choice} ({USER_LANGUAGE})\n")

    # --- Show Initial Recommendations ---
    recommendations = recommender.get_initial_questions() # Use the method to set initial state
    header = translate_text("💡 Recommended questions: (Type 'back' to return to the previous list)", target_lang=USER_LANGUAGE)
    print(header)
    for i, q in enumerate(recommendations, 1):
        q_translated = translate_text(q, target_lang=USER_LANGUAGE)
        print(f"{i}. {q_translated}")

    # --- Main Chat Loop ---
    while True:
        prompt = translate_text("\nYou:", target_lang=USER_LANGUAGE)
        user_input = input(prompt + " ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            goodbye_msg = translate_text("👋 Goodbye!", target_lang=USER_LANGUAGE)
            print(goodbye_msg)
            break

        # --- NEW: Handle the 'back' command ---
        if user_input.lower() == "back":
            print(translate_text("↩️ Going back...", target_lang=USER_LANGUAGE))
            recommendations = recommender.go_back()
            # Display the previous recommendations
            header = translate_text("\n💡 Recommended questions:", target_lang=USER_LANGUAGE)
            print(header)
            for i, q in enumerate(recommendations, 1):
                q_translated = translate_text(q, target_lang=USER_LANGUAGE)
                print(f"{i}. {q_translated}")
            continue # Skip the rest of the loop and wait for new input

        query = ""

        is_numeric_selection = user_input.isdigit() and 1 <= int(user_input) <= len(recommendations)
        
        if is_numeric_selection:
            selected_index = int(user_input) - 1
            query = recommendations[selected_index]
            confirmation_text = f"✅ You selected: {translate_text(query, USER_LANGUAGE)}"
            print(confirmation_text)
        else:
            query = user_input
            if USER_LANGUAGE != "en":
                query = translate_text(query, target_lang="en", source_lang=USER_LANGUAGE)

        # --- Get and display answer (This part remains the same) ---
        results = bot.search(query, top_k=5, alpha=0.8)
        if not results:
            no_answer_msg = translate_text("I'm sorry, I couldn't find an answer to that.", target_lang=USER_LANGUAGE)
            print(f"Bot: {no_answer_msg}")
            continue

        answer_en = results[0]['answer']
        answer_translated = translate_text(answer_en, target_lang=USER_LANGUAGE)
        print(f"Bot: {answer_translated}")

        # --- Get and display the next set of recommendations ---
        recommendations = recommender.recommend(query)
        next_header = translate_text("\n💡 Next recommendations:", target_lang=USER_LANGUAGE)
        print(next_header)
        for i, q in enumerate(recommendations, 1):
            q_translated = translate_text(q, target_lang=USER_LANGUAGE)
            print(f"{i}. {q_translated}")