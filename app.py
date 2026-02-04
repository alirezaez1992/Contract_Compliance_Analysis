# %%writefile app.py      This is for running on Colab             
import json
import streamlit as st
import pandas as pd

from pipeline import analyze_pdf_bytes

import subprocess, re, time, textwrap


st.set_page_config(page_title="Compliance Analyzer", layout="wide")
st.title("Contract Compliance Analyzer (Colab Demo)")

st.write("Upload a PDF contract. The app parses it, runs GenAI compliance analysis, and returns structured JSON.")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    st.success("PDF uploaded.")
    pdf_bytes = uploaded.read()

    if st.button("Run analysis"):
        status = st.status("Starting...", expanded=True)
        try:
            status.update(label="Running analysis (PDF parsing → embeddings → LLM)…", state="running")
            report = analyze_pdf_bytes(pdf_bytes)
            status.update(label="Done.", state="complete")

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
                "Download JSON",
                data=json.dumps(report, indent=2).encode("utf-8"),
                file_name="compliance_report.json",
                mime="application/json",
            )
        except Exception as e:
            status.update(label="Error", state="error")
            st.exception(e)
else:
    st.info("Upload a PDF to begin.")


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

