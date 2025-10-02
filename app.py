from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, re

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def detect_plagiarism(query, corpus, doc_threshold=0.35, sentence_threshold=0.5):
    vectorizer = TfidfVectorizer().fit([query] + corpus)
    tfidf = vectorizer.transform([query] + corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    # Document-level similarity
    doc_report = [{"id": i, "similarity": round(float(s),4)} for i, s in enumerate(sims)]

    # Sentence-level matches
    q_sents = split_into_sentences(query)
    all_sents, sent_map = [], []
    for i, doc in enumerate(corpus):
        for s in split_into_sentences(doc):
            all_sents.append(s)
            sent_map.append((i, s))
    matches = []
    if all_sents:
        tfidf_sent = TfidfVectorizer().fit(q_sents + all_sents)
        q_tfidf = tfidf_sent.transform(q_sents)
        c_tfidf = tfidf_sent.transform(all_sents)
        sim_matrix = cosine_similarity(q_tfidf, c_tfidf)
        for qi, qsent in enumerate(q_sents):
            best_idx = sim_matrix[qi].argmax()
            score = float(sim_matrix[qi].max())
            if score >= sentence_threshold:
                doc_id, match_sent = sent_map[best_idx]
                matches.append({
                    "query_sentence": qsent,
                    "doc_id": doc_id,
                    "matched_sentence": match_sent,
                    "score": round(score,4)
                })
    return doc_report, matches

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query_file = request.files["query"]
        refs = request.files.getlist("references")

        query_text = query_file.read().decode("utf-8")
        corpus = [f.read().decode("utf-8") for f in refs if f]

        doc_report, sent_matches = detect_plagiarism(query_text, corpus)
        return render_template("result.html", docs=doc_report, matches=sent_matches)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
