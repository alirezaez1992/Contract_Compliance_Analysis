#%%writefile app.py  For running on Colab
import json
import streamlit as st
import pandas as pd

from pipeline_chat import analyze_pdf_bytes, chat_with_pdf_bytes
import subprocess, re, time, textwrap


st.set_page_config(page_title="PDF Chat + Compliance Analyzer", layout="wide")
st.title("Contract Compliance Analyzer + Chat (Colab Demo)")
st.write(
    "Upload a PDF contract. You can either (1) run the predefined compliance checklist, "
    "or (2) chat with the PDF by asking any question. Answers are grounded with verbatim quotes + page numbers."
)

# -----------------------------
# Session state
# -----------------------------
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "chat_history" not in st.session_state:
    # pipeline expects list of {"role": "...", "content": "..."}
    st.session_state.chat_history = []
if "last_compliance_report" not in st.session_state:
    st.session_state.last_compliance_report = None

# -----------------------------
# Sidebar: upload + mode
# -----------------------------
with st.sidebar:
    st.header("PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded:
        st.session_state.pdf_bytes = uploaded.read()
        st.session_state.pdf_name = uploaded.name
        st.success(f"Loaded: {uploaded.name}")
    else:
        st.info("Upload a PDF to begin.")

    st.divider()
    mode = st.radio(
        "Mode",
        ["Chat with PDF", "Compliance Checklist"],
        index=0
    )

    st.divider()
    if st.button("Reset chat", use_container_width=True):
        st.session_state.chat_history = []
        st.toast("Chat cleared.")

# -----------------------------
# Main UI
# -----------------------------
pdf_bytes = st.session_state.pdf_bytes

if not pdf_bytes:
    st.stop()

# Top info row
col1, col2 = st.columns([2, 1])
with col1:
    st.caption(f"Active PDF: **{st.session_state.pdf_name or 'uploaded.pdf'}**")
with col2:
    st.caption("Tip: Ask specific questions like: *Does it mention TLS 1.2?*")

# -----------------------------
# Mode 1: Chat
# -----------------------------
if mode == "Chat with PDF":
    st.subheader("Chat with the PDF")

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_q = st.chat_input("Ask a question about the PDF…")
    if user_q:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.write(user_q)

        with st.chat_message("assistant"):
            status = st.status("Thinking…", expanded=False)
            try:
                status.update(label="Retrieving evidence + generating answer…", state="running")
                ans = chat_with_pdf_bytes(
                    pdf_bytes,
                    user_q,
                    chat_history=st.session_state.chat_history
                )
                status.update(label="Done.", state="complete")

                # Display answer
                st.write(ans["answer"])
                st.caption(f"Confidence: {ans.get('confidence', 0.0):.2f}")

                # Quotes
                if ans.get("relevant_quotes"):
                    with st.expander("Evidence (verbatim quotes)", expanded=True):
                        for q in ans["relevant_quotes"]:
                            st.markdown(f"- {q}")

                # Show raw JSON for debugging / transparency
                with st.expander("Raw JSON output"):
                    st.json(ans)

                # Add assistant message to history (store the main answer text)
                st.session_state.chat_history.append({"role": "assistant", "content": ans["answer"]})

            except Exception as e:
                status.update(label="Error", state="error")
                st.exception(e)

    st.divider()
    st.subheader("Download chat log")
    st.download_button(
        "Download chat history (JSON)",
        data=json.dumps(
            {"pdf": st.session_state.pdf_name, "history": st.session_state.chat_history},
            indent=2
        ).encode("utf-8"),
        file_name="pdf_chat_history.json",
        mime="application/json",
        use_container_width=True,
    )

# -----------------------------
# Mode 2: Compliance checklist
# -----------------------------
else:
    st.subheader("Predefined Compliance Checklist")

    run = st.button("Run analysis", type="primary")
    if run:
        status = st.status("Starting…", expanded=True)
        try:
            status.update(label="Running analysis (PDF parsing → embeddings → LLM)…", state="running")
            report = analyze_pdf_bytes(pdf_bytes)  # compliance mode
            st.session_state.last_compliance_report = report
            status.update(label="Done.", state="complete")
        except Exception as e:
            status.update(label="Error", state="error")
            st.exception(e)

    report = st.session_state.last_compliance_report
    if report:
        st.subheader("Structured compliance JSON")
        st.json(report)

        st.subheader("Readable table")
        df = pd.DataFrame([{
            "Compliance Question": it["compliance_question"],
            "Compliance State": it["compliance_state"],
            "Confidence": it["confidence"],
            "Relevant Quotes": "\n".join(it["relevant_quotes"]),
            "Rationale": it["rationale"]
        } for it in report["items"]])
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download compliance JSON",
            data=json.dumps(report, indent=2).encode("utf-8"),
            file_name="compliance_report.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.info("Click **Run analysis** to generate the compliance report.")


##Cloud 

# Start cloudflared and capture its logs live (no logfile needed)
proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8501"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

public_url = None
deadline = time.time() + 30  # wait up to ~30s for the URL

log_lines = []
while time.time() < deadline and public_url is None:
    line = proc.stdout.readline()
    if not line:
        time.sleep(0.1)
        continue
    log_lines.append(line.rstrip())

    m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
    if m:
        public_url = m.group(0)
        break

if public_url:
    print("\n" + "="*70)
    print("✅ Streamlit GUI is live! Open this URL in your browser:\n")
    print(public_url)
    print("="*70 + "\n")
else:
    print("\n❌ Could not find the trycloudflare URL in time.")
    print("Last logs:\n")
    print("\n".join(log_lines[-25:]))
    print("\nTroubleshooting:")
    print("1) Make sure Streamlit is running on port 8501.")
    print("2) Re-run the Streamlit cell, then re-run this cell.")
