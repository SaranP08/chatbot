import json
import random
from transformers import pipeline

# --- 1. Load base Q&A dataset ---
with open("data/qa_pairs.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# --- 2. Initialize paraphraser ---
paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

# --- 3. Define synonyms for domain-specific words ---
synonyms = {
    "see": ["view", "check", "look at"],
    "access": ["open", "retrieve", "get"],
    "calendar": ["schedule", "planner"],
    "crop": ["farming", "agriculture"]
}

def synonym_variants(question):
    variants = [question]
    for word, replacements in synonyms.items():
        if word in question.lower():
            for r in replacements:
                variants.append(question.replace(word, r))
    return list(set(variants))  # remove duplicates

def generate_paraphrases(question, num_return=3):
    prompt = f"paraphrase: {question}"
    outputs = paraphraser(
        prompt,
        num_return_sequences=num_return,
        max_length=64,
        clean_up_tokenization_spaces=True
    )
    return [o["generated_text"] for o in outputs]

# --- 4. Expand dataset ---
expanded_data = []
for item in qa_data:
    base_q = item["question"]
    answer = item["answer"]

    # original question
    variants = [base_q]

    # add synonym-based variants
    variants.extend(synonym_variants(base_q))

    # add paraphrased variants
    try:
        variants.extend(generate_paraphrases(base_q))
    except Exception as e:
        print(f"Paraphrasing failed for: {base_q} -> {e}")

    # save all unique variants mapped to same answer
    for v in set(variants):
        expanded_data.append({"question": v.strip(), "answer": answer})

print(f"Original Q&A: {len(qa_data)} → Expanded Q&A: {len(expanded_data)}")

# --- 5. Save expanded dataset ---
with open("data/qa_pairs_expanded.json", "w", encoding="utf-8") as f:
    json.dump(expanded_data, f, indent=2, ensure_ascii=False)

print("✅ Expanded dataset saved to data/qa_pairs_expanded.json")
