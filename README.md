# Clinical_Entities_In_Radilogy_report
# 🧠 Clinical Entity Extraction in Radiology Reports

This Streamlit-based application enables automated extraction and analysis of clinical entities from radiology reports using state-of-the-art biomedical NLP models and Azure OpenAI (`gpt-4o`). It is designed for healthcare professionals to identify key information such as diagnoses, symptoms, anatomical locations, and treatment suggestions from unstructured medical reports.

---

## 🚀 Features

- 📄 Upload support for **PDF, DOCX, and TXT** files
- 🧹 Intelligent **noise removal and text preprocessing**
- 🤖 **Named Entity Recognition (NER)** using the `d4data/biomedical-ner-all` model
- 🌈 **Entity highlighting**, categorization, and tabular view
- 📊 Summary and **download options (CSV/JSON)** for extracted entities
- 🩺 **Disease & treatment recommendations** using **Azure OpenAI (`gpt-4o`)**
- 🧠 Contextual **diagnosis-symptom-location** mapping
- 🎯 Interactive UI for healthcare professionals and radiologists

---

## 🛠️ Tech Stack

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)
- `pdfplumber`, `docx2txt`, `pydicom`, `pandas`, `re`, `requests`, etc.

---

## 🏁 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/radiology-entity-extraction.git
cd radiology-entity-extraction
