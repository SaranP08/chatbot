from rank import HybridChatBot
from deep_translator import GoogleTranslator

# User selects preferred language

indian_languages = {
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
    "English": "en",
    "french": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
}


def translate_text(text, target_lang, source_lang="auto"):
    """Helper to translate text."""
    return GoogleTranslator(source=source_lang, target=target_lang).translate(text)

if __name__ == "__main__":
    bot = HybridChatBot()

    language = input("select your language:")

    USER_LANGUAGE = indian_languages.get(language)

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        # Step 1: Translate query to English (bot's working language)
        if USER_LANGUAGE != 'en':
            query = translate_text(query, target_lang="en")

        # Step 2: Search bot with English query
        results = bot.search(query, top_k=5, alpha=0.8)
        best = results[0]
        answer = best['answer']

        # Step 3: Translate answer back to user’s language
        if USER_LANGUAGE != 'en':
            answer = translate_text(answer, target_lang=USER_LANGUAGE)

        print(f"Bot: {answer}")
