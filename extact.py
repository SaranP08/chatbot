import pdfplumber
import re
import json

def extract_qa_from_pdf(pdf_path, output_json="data/qa_pairs.json"):
    qa_pairs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            # Adjust regex based on your PDF formatting
            matches = re.findall(r"Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)", text, re.S)
            qa_pairs.extend(matches)

    qa_data = [{"question": q.strip(), "answer": a.strip()} for q, a in qa_pairs]

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(qa_data)} Q&A pairs → {output_json}")

if __name__ == "__main__":
    extract_qa_from_pdf("data/SatyuktQueries.pdf")
