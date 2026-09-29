import asyncio
from pathlib import Path
import time
import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

# ✅ FIX 1: Corrected argument names (singular, not plural)
st.set_page_config(
    page_title="RAG Ingest PDF",
    page_icon="📄",
    layout="centered"
)


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name

    # ✅ FIX 2: Actually write the bytes to the file!
    # The previous code got the buffer but didn't save it.
    file_path.write_bytes(file.getbuffer())

    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",  # ✅ FIX 3: changed 'names' to 'name'
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and Triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        asyncio.run(send_rag_ingest_event(path))
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {path.name}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    # client.send returns a list of IDs, we take the first one
    ids = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return ids[0]


def _inngest_api_base() -> str:
    # Ensure this points to your local Inngest Dev Server
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        # Prevent UI crash if Dev Server is offline
        st.error(f"Could not connect to Inngest Dev Server: {e}")
        return []


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 1.0) -> dict:
    start = time.time()
    last_status = None

    progress_bar = st.progress(0, text="Waiting for AI...")

    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status

            if status in ("Completed", "Succeeded", "Success", "Finished"):
                progress_bar.empty()
                return run.get("output") or {}

            if status in ("Failed", "Cancelled"):
                progress_bar.empty()
                raise RuntimeError(f"Function run {status}: {run.get('output')}")

            # Simple visual feedback
            if status == "Running":
                progress_bar.progress(50, text="AI is thinking...")

        if time.time() - start > timeout_s:
            progress_bar.empty()
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")

        time.sleep(poll_interval_s)


with st.form("rag_query_form"):
    question_input = st.text_input("Your question")
    top_k_input = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=3, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question_input.strip():
        with st.spinner("Sending event and generating answer..."):
            try:
                # 1. Send Event
                event_id = asyncio.run(send_rag_query_event(question_input.strip(), int(top_k_input)))

                # 2. Poll for Result
                output = wait_for_run_output(event_id)

                answer = output.get("answer", "")
                sources = output.get("sources", [])

                st.subheader("Answer")
                st.write(answer or "(no answer)")

                if sources:
                    st.write("---")
                    st.subheader("Sources")
                    for s in sources:
                        st.markdown(f"- `{s}`")

            except Exception as e:
                st.error(f"Error: {e}")