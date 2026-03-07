import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plagiarism Checker",
    page_icon="🔍",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}

/* Card */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 18px;
    backdrop-filter: blur(10px);
}

/* Similarity badge */
.badge-high   { background:#ff4e4e33; border:1px solid #ff4e4e; border-radius:8px; padding:3px 10px; color:#ff6b6b; font-weight:600; }
.badge-medium { background:#ffb34733; border:1px solid #ffb347; border-radius:8px; padding:3px 10px; color:#ffb347; font-weight:600; }
.badge-low    { background:#43e97b33; border:1px solid #43e97b; border-radius:8px; padding:3px 10px; color:#43e97b; font-weight:600; }

/* Title */
h1 { background: linear-gradient(90deg, #a18cd1, #fbc2eb); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

/* Progress bar track */
.stProgress > div > div > div > div { background: linear-gradient(90deg, #a18cd1, #fbc2eb) !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 1px dashed rgba(161,140,209,0.5);
    border-radius: 12px;
    padding: 6px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #a18cd1, #fbc2eb);
    color: #1a1a2e;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 10px 30px;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(161,140,209,0.4);
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def split_into_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def detect_plagiarism(query: str, corpus: list[str],
                      doc_threshold: float = 0.35,
                      sentence_threshold: float = 0.5):
    vectorizer = TfidfVectorizer().fit([query] + corpus)
    tfidf = vectorizer.transform([query] + corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    doc_report = [{"id": i, "similarity": round(float(s), 4)}
                  for i, s in enumerate(sims)]

    q_sents = split_into_sentences(query)
    all_sents, sent_map = [], []
    for i, doc in enumerate(corpus):
        for s in split_into_sentences(doc):
            all_sents.append(s)
            sent_map.append((i, s))

    matches = []
    if all_sents and q_sents:
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
                    "score": round(score, 4),
                })

    return doc_report, matches


def badge(sim: float) -> str:
    if sim >= 0.65:
        return f'<span class="badge-high">🔴 {sim*100:.1f}%</span>'
    elif sim >= 0.35:
        return f'<span class="badge-medium">🟡 {sim*100:.1f}%</span>'
    else:
        return f'<span class="badge-low">🟢 {sim*100:.1f}%</span>'


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;font-size:2.6rem;'>🔍 Plagiarism Checker</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#aaa;margin-top:-10px;'>Upload your document and references — we'll do the rest.</p>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([1, 1.6], gap="large")

with col1:
    st.markdown("### 📄 Your Document")
    query_file = st.file_uploader("Upload the document to check", type=["txt"], key="query")

    st.markdown("### 📚 Reference Documents")
    ref_files = st.file_uploader("Upload one or more reference files", type=["txt"],
                                  accept_multiple_files=True, key="refs")

    st.markdown("### ⚙️ Thresholds")
    doc_thresh  = st.slider("Document similarity threshold",  0.0, 1.0, 0.35, 0.05)
    sent_thresh = st.slider("Sentence similarity threshold", 0.0, 1.0, 0.50, 0.05)

    run = st.button("🚀 Check for Plagiarism", use_container_width=True)

with col2:
    if run:
        if not query_file:
            st.error("⚠️ Please upload the document you want to check.")
        elif not ref_files:
            st.error("⚠️ Please upload at least one reference document.")
        else:
            query_text = query_file.read().decode("utf-8", errors="ignore")
            corpus     = [f.read().decode("utf-8", errors="ignore") for f in ref_files]
            names      = [f.name for f in ref_files]

            with st.spinner("Analysing…"):
                doc_report, sent_matches = detect_plagiarism(
                    query_text, corpus, doc_thresh, sent_thresh)

            # ── Overall score ──────────────────────────────────────────────
            max_sim = max(r["similarity"] for r in doc_report) if doc_report else 0.0
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <p style="font-size:1rem;color:#aaa;margin-bottom:4px;">Overall max similarity</p>
                <p style="font-size:3rem;font-weight:700;margin:0;">
                    {max_sim*100:.1f}%
                </p>
                {badge(max_sim)}
            </div>""", unsafe_allow_html=True)

            st.markdown("### 📊 Document-level Results")
            for r in doc_report:
                name = names[r["id"]] if r["id"] < len(names) else f"Doc {r['id']}"
                sim  = r["similarity"]
                with st.expander(f"**{name}** — {sim*100:.1f}% similar"):
                    st.progress(sim)
                    st.markdown(badge(sim), unsafe_allow_html=True)

            st.divider()
            st.markdown("### 🔎 Sentence-level Matches")
            if sent_matches:
                for m in sent_matches:
                    ref_name = names[m["doc_id"]] if m["doc_id"] < len(names) else f"Doc {m['doc_id']}"
                    with st.expander(f"Score {m['score']*100:.1f}% — matched in **{ref_name}**"):
                        c1, c2 = st.columns(2)
                        c1.markdown("**Your sentence**")
                        c1.info(m["query_sentence"])
                        c2.markdown("**Matched sentence**")
                        c2.warning(m["matched_sentence"])
            else:
                st.success("✅ No significant sentence-level matches found above threshold.")
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:60px 30px;">
            <p style="font-size:3rem;">📂</p>
            <p style="font-size:1.1rem;color:#aaa;">Upload your files on the left<br>and hit <strong>Check for Plagiarism</strong>.</p>
        </div>""", unsafe_allow_html=True)
