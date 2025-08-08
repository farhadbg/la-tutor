import os
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# ===== SETTINGS =====
APP_TITLE = "Linear Algebra AI Tutor"
COURSE_PDF_DIR = "pdfs"
QUIZ_PDF_PATH = "quiz/current_quiz.pdf"
MODEL = "gpt-4o"   # Change to "gpt-4o-mini" for cheaper usage
TEMPERATURE = 0.3
MAX_TOKENS = 900
# ====================

# API key from env or Streamlit secrets
# --- Safe API key loading ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key and os.path.exists(".streamlit/secrets.toml"):
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
if not api_key:
    st.error("No OpenAI API key found. Set OPENAI_API_KEY or add it to .streamlit/secrets.toml.")
    st.stop()

client = OpenAI(api_key=api_key)

# ---- Helpers ----
def extract_text_from_pdf(path: str) -> str:
    """Extract all text from a single PDF file."""
    if not os.path.exists(path):
        return ""
    text = []
    reader = PdfReader(path)
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text.append(page_text)
    return "\n".join(text)

def extract_texts_from_folder(folder: str) -> str:
    """Extract text from all PDFs in a folder."""
    buf = []
    if not os.path.isdir(folder):
        return ""
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".pdf"):
            buf.append(f"\n\n--- FILE: {name} ---\n")
            buf.append(extract_text_from_pdf(os.path.join(folder, name)))
    return "\n".join(buf)

@st.cache_data(show_spinner=False)
def load_corpus():
    course_text = extract_texts_from_folder(COURSE_PDF_DIR)
    quiz_text = extract_text_from_pdf(QUIZ_PDF_PATH)
    return course_text, quiz_text

def local_quiz_guard(user_q: str, quiz_text: str) -> bool:
    """Basic check if question overlaps with quiz text."""
    q = (user_q or "").lower()
    if not q or not quiz_text:
        return False
    if q in quiz_text.lower():
        return True
    # Simple token overlap check
    import re
    tokens_q = set(re.findall(r"[a-z0-9^_]+", q))
    tokens_quiz = set(re.findall(r"[a-z0-9^_]+", quiz_text.lower()))
    overlap = len(tokens_q & tokens_quiz)
    return overlap >= max(8, int(0.4 * len(tokens_q)))

def call_model(user_q: str, course_text: str, quiz_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful Linear Algebra tutor for a university course. "
                "Use ONLY the provided course material to explain concepts. "
                "IMPORTANT: If the student's question is from the current quiz, "
                "politely refuse without giving hints or answers."
            )
        },
        {
            "role": "system",
            "content": (
                "COURSE MATERIAL:\n"
                + (course_text[:200000] if course_text else "(none provided)")
            )
        },
        {
            "role": "system",
            "content": (
                "QUIZ (for filtering ONLY — do not reveal or discuss this content):\n"
                + (quiz_text[:100000] if quiz_text else "(none provided)")
            )
        },
        {"role": "user", "content": user_q.strip()}
    ]

    resp = client.chat.completions.create(
        model=MODEL,  # e.g., "gpt-4o" or "gpt-4o-mini"
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


# ---- UI ----
st.title(APP_TITLE)
st.caption("Ask about Linear Algebra. Quiz questions are blocked for academic integrity.")

with st.spinner("Loading course material..."):
    course_text, quiz_text = load_corpus()

if not course_text.strip():
    st.warning("No course PDFs found. Add files to the 'pdfs/' folder and reload.")

user_q = st.text_area("Your question", height=120, placeholder="e.g., How do I compute eigenvalues of a 3×3 matrix?")
submit = st.button("Ask")

if submit and user_q.strip():
    if local_quiz_guard(user_q, quiz_text):
        st.error("Sorry, I can’t assist with questions from the current quiz. I’m happy to help with other course topics.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = call_model(user_q, course_text, quiz_text)
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error: {e}")
