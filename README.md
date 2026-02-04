# Alireza Esmaeilzehi
## Contract Compliance Analysis using Generative AI

### Documentation
Detailed information about the project is provided in `Doc.pdf`. Please refer to it for more details.

All the developments have been conducted on Google Colab with NVIDIA A100 (40 GB) GPU. 

### Backend 

Install dependencies with `pip`:

```bash
pip install -r requirements.txt
```
The pipeline implementations, including, preprocessing, information retrieval, LLM reasoning, and JASON output generation are located:
```bash
python pipeline.py
```

For chat functionality:
```bash
python pipeline_chat.py
```
### Frontend  

The frontend and user-interface implementations with Streamlit are located:
```bash
python app.py
```

For chat functionality:
```bash
python app_chat.py
```

### Structured Output

The example of structured output is located:
```bash
compliance_report.json
```

For chat functionality:
```bash
pdf_chat_history.json
```




 

