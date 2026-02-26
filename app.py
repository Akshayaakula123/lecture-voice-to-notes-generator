import streamlit as st
import whisper
import tempfile
from transformers import pipeline
from fpdf import FPDF
import torch

st.set_page_config(page_title="AI Lecture Notes Generator")

st.title("🎙 Lecture Voice-to-Notes Generator")
st.write("Upload your lecture audio and convert it into smart study notes.")

uploaded_file = st.file_uploader("Upload Lecture Audio File", type=["mp3", "wav", "m4a"])

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-6-6",
        framework="pt"
    )

@st.cache_resource
def load_generator():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        framework="pt"
    )

if uploaded_file is not None:

    st.success("Audio file uploaded successfully!")

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    # ---------------- TRANSCRIPTION ----------------
    st.info("Transcribing audio... Please wait ⏳")

    whisper_model = load_whisper()
    result = whisper_model.transcribe(temp_audio_path)
    transcript = result["text"]

    st.subheader("📝 Transcription")
    st.write(transcript)

    trimmed_text = transcript[:2000]

    # ---------------- SUMMARY (STUDY NOTES) ----------------
    st.info("Generating study notes... Please wait ⏳")

    summarizer = load_summarizer()
    summary_output = summarizer(trimmed_text, max_new_tokens=120, do_sample=False)
    summary_text = summary_output[0]["summary_text"]

    st.subheader("📚 Study Notes")
    st.write(summary_text)

    # ---------------- PRACTICE QUESTIONS ----------------
    st.info("Generating practice questions... Please wait ⏳")

    generator = load_generator()

    concept_prompt = f"""
Extract exactly 5 important short concepts from this lecture.
Return them separated by comma only.

Lecture:
{trimmed_text}
"""

    concept_output = generator(
        concept_prompt,
        max_new_tokens=60,
        do_sample=False
    )

    concepts_raw = concept_output[0]["generated_text"]
    concepts = list(dict.fromkeys([c.strip() for c in concepts_raw.split(",") if c.strip()]))

    while len(concepts) < 5:
        concepts.append("Artificial Intelligence")

    question_patterns = [
        "What is {concept}?",
        "Explain the importance of {concept}.",
        "How does {concept} work?",
        "What are the applications of {concept}?",
        "Why is {concept} important?"
    ]

    questions = []

    for i in range(5):
        concept = concepts[i]
        question_text = question_patterns[i].format(concept=concept)

        answer_prompt = f"""
Give a short 2-line clear answer for this question based on the lecture.

Question: {question_text}

Lecture:
{trimmed_text}
"""

        answer_output = generator(
            answer_prompt,
            max_new_tokens=80,
            do_sample=False
        )

        answer_text = answer_output[0]["generated_text"].strip()

        full_q = f"{i+1}. Question: {question_text}\nAnswer: {answer_text}"
        questions.append(full_q)

    questions_text = "\n\n".join(questions)

    st.subheader("🧠 Practice Questions with Answers")
    st.write(questions_text)

    # ---------------- PDF DOWNLOAD ----------------
    st.info("Preparing PDF... 📄")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Arial", size=11)

    pdf.multi_cell(0, 8, "Lecture Voice-to-Notes Report\n\n")
    pdf.multi_cell(0, 8, "Transcription:\n" + transcript + "\n\n")
    pdf.multi_cell(0, 8, "Study Notes:\n" + summary_text + "\n\n")
    pdf.multi_cell(0, 8, "Practice Questions:\n" + questions_text + "\n\n")

    pdf_file = "Lecture_Notes.pdf"
    pdf.output(pdf_file)

    with open(pdf_file, "rb") as f:
        st.download_button(
            "📥 Download Notes as PDF",
            f,
            file_name="Lecture_Notes.pdf",
            mime="application/pdf"
        )