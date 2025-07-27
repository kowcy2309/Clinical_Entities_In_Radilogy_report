import pandas as pd
import streamlit as st
import pdfplumber
import docx2txt
import pydicom
from pydicom.errors import InvalidDicomError
from io import StringIO
import re
import string
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import openai
import requests

model_name = "d4data/biomedical-ner-all"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
unwanted_labels = {"UnwantedEntityLabel1", "UnwantedEntityLabel2"}

llama_api_key = "MGIVN9XU9u9bf7IUFcHRdMUn5lmjJ1XF"
llama_api_base = "https://meta-llama-3-1-405b-instruct-pwu.eastus.models.ai.azure.com/v1/chat/completions"

st.set_page_config(layout="wide")

logo_path = "https://trigent.com/wp-content/uploads/Trigent_Axlr8_Labs.png"
st.markdown(
    f"""
    <div style="text-align:center;">
        <img src="{logo_path}" alt="Trigent Logo" style="max-width:100%;">
    </div>
    """,
    unsafe_allow_html=True
)
st.header("Clinical Entity Extraction In Radiology Reports", divider='rainbow')

st.write("""
This Streamlit application automates the extraction and classification of clinical entities from radiology reports. Designed for radiologists and healthcare providers, it identifies diagnoses, anatomical findings, and treatment recommendations. Users can upload reports in various formats, and the app uses advanced NLP models to extract and classify clinical information. The results are displayed interactively and can be downloaded as structured data.
""")

import openai

# Azure OpenAI configuration
openai.api_type = "azure"
openai.api_key = "51ba5d46601c477b844d3883af93463c"
openai.api_base = "https://genai-trigent-openai.openai.azure.com/"
openai.api_version = "2024-02-15-preview"
deployment_name = "gpt-4o"

def analyze_clinical_entities_azure(entities):
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a medical expert specializing in identifying diseases and suggesting treatments based on clinical data."
            },
            {
                "role": "user",
                "content": f"Here are the extracted clinical entities: {entities}. Based on this, please suggest possible diseases and appropriate treatment options."
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4",  # required, but ignored by Azure
            deployment_id=deployment_name,
            messages=messages,
            temperature=0.7,
            max_tokens=700
        )

        return response['choices'][0]['message']['content']
    
    except Exception as e:
        return f"Error calling Azure OpenAI: {e}"

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"  
    return text

def extract_text_from_docx(docx_file):
    return docx2txt.process(docx_file)

def extract_text_from_txt(txt_file):
    return txt_file.read().decode('utf-8')

def preprocess_text(text):
    lines = text.splitlines()
    cleaned_lines = []

    noise_patterns = [
        re.compile(r'(?i)^(?:phone\s*[:\s]*|phone\s*number\s*[:\s]*|contact\s*[:\s]*|tel\s*[:\s]*|telephone\s*[:\s]*|'
                   r'date\s*[:\s]*|date\s*of\s*birth\s*[:\s]*|reported\s*date\s*[:\s]*|recorded\s*date\s*[:\s]*|'
                   r'patient\s*[:\s]*|subject\s*[:\s]*|id\s*[:\s]*|info\s*[:\s]*|'
                   r'[\+\d\s.-]{10,}|\d{1,2}[-.\s/]\d{1,2}[-.\s/]\d{2,4}|[X]{2,}|(?:Dr\.|Mr\.|Mrs\.|Ms\.|'
                   r'Prof\.|Dr\s|Mr\s|Mrs\s|Ms\s|Prof\s)[\sX]+.*|(?:Referred\s*by\s*[:\s]*)(?:Dr\.|Mr\.|Mrs\.|Ms\.|'
                   r'Prof\.|Dr\s|Mr\s|Mrs\s|Ms\s|Prof\s)[\sX]+).*$',  
                   re.MULTILINE | re.IGNORECASE),
        re.compile(r'\bPage\s+\d+\s+of\s+\d+\b', re.IGNORECASE),
        re.compile(r'\bDOB\s*[:\s]*\d{2}/\d{2}/\d{4}\b', re.IGNORECASE),
        re.compile(r'\b\d{1,5}\s[\w\s.,]+,\sSuite\s*\d+\s[\w\s.,]+\s\w{2}\s\d{5}\b', re.IGNORECASE),
        re.compile(r'\b(?:Phone|Fax|FAX|Telephone|Tel):?\s*\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b', re.IGNORECASE),
        re.compile(r'\b[A-Z][A-Z\s]+,?\s*\d{1,5}\s[\w\s.,]+,\s(?:Suite\s*\d+|Building\s*\d+)?\s[\w\s.,]+\s\w{2}\s\d{5}-\d{4}\b', re.IGNORECASE),
        re.compile(r'\b(?:Referring\s+Physician|Consulting\s+Physician|Attending\s+Physician|Dr\.|Dr\s|'
                   r'Physician|Ref\.)\s*[:\s]*\s*Dr\.\s*[A-Z]{5,}\b', re.IGNORECASE),
        re.compile(r'-\s*Electronically\s*Signed\s*by:\s*[A-Z\s]+,\s*[A-Z\s]+ on \d{2}/\d{2}/\d{4}\s\d{1,2}:\d{2}:\d{2}\s[APM]{2}', re.IGNORECASE),
        re.compile(r'\bFAX:\s*\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b', re.IGNORECASE),
        re.compile(r'^\bM\.App\.Sc\.,\s*D\.C\.,\s*Ph\.D\.,\s*D\.A\.C\.B\.R\.\b$', re.IGNORECASE),
        re.compile(r'THIS REPORT WAS ELECTRONICALLY SIGNED.*$', re.IGNORECASE | re.DOTALL),
        re.compile(r'Report approved on.*$', re.IGNORECASE | re.DOTALL),
        re.compile(r'NationalRad\s*\|\s*Headquartered:.*$', re.IGNORECASE | re.DOTALL),
        re.compile(r'\[ NationalRad Sample Musculoskeletal Radiology Report \].*$', re.IGNORECASE | re.DOTALL),
        re.compile(r'Imaging Center.*$', re.IGNORECASE | re.DOTALL),
        re.compile(r'^\d{1,5}\s[\w\s.,]+,\s[^\d].*\d{5}$', re.IGNORECASE | re.DOTALL),
        re.compile(r'^\bDOB:\s*\d{1,2}/\d{1,2}/\d{4}\b$', re.IGNORECASE | re.DOTALL),
        re.compile(r'FILE\s*#:\s*\d+$', re.IGNORECASE),
        re.compile(r'\d{1,5}\s[\w\s.,]+,\s[^\d].*\d{5}$', re.IGNORECASE | re.DOTALL),
        re.compile(r'\b\d{1,5}\s[\w\s.,]+,\s(?:[A-Za-z]+\s+){1,2}[A-Z]{2}\s\d{5}\b', re.IGNORECASE),
        re.compile(r'\[.*?\]', re.IGNORECASE)
    ]

    for line in text.splitlines():
        cleaned_line = line
        for pattern in noise_patterns:
            cleaned_line = pattern.sub('', cleaned_line).strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines)


def categorize_entity(entity):
    category_mapping = {
        "Disease_disorder": "Diagnosis",
        "Diagnostic_procedure": "Diagnosis",
        "Sign_symptom": "Symptom",
        "Biological_structure": "Anatomical Location",
        "Therapeutic_procedure": "Treatment Recommendation",
        "Lab_value": "Lab Value",
        "Detailed_description": "Detailed Description",
        "Area": "Area"
    }
    return category_mapping.get(entity, "Other")

def extract_and_format_clinical_entities(text):
    entities = nlp(text)
    detailed_entities = []

    for entity in entities:
        word = entity['word']
        label = entity['entity_group']
        score = entity['score']
        start = entity['start']
        end = entity['end']
        
        if len(word) <= 2 or word[0] in string.punctuation:
            continue
        
        if label in unwanted_labels:
            continue
        
        if score >= 0.7:
            category = categorize_entity(label)
            detailed_entities.append({
                "Entity": word,
                "Label": label,
                "Score": score,
                "Start": start,
                "End": end,
                "Category": category
            })

    unique_entities = {}
    for entity in detailed_entities:
        cat = entity["Category"]
        word = entity["Entity"]
        
        if cat not in unique_entities:
            unique_entities[cat] = set()
        
        if word.lower() not in map(str.lower, unique_entities[cat]):
            unique_entities[cat].add(word)

    unique_entities = {cat: list(entities) for cat, entities in unique_entities.items()}

    return detailed_entities, unique_entities

def generate_detailed_descriptions(entities):
    entity_descriptions = {}

    for category, entity_list in entities.items():
        if category not in entity_descriptions:
            entity_descriptions[category] = []
        entity_descriptions[category].extend(entity_list)

    return entity_descriptions

def analyze_clinical_entities_llama(entities, llama_api_key):
    headers = {
        "Authorization": f"Bearer {llama_api_key}",
        "Content-Type": "application/json"
    }
    messages = [
        {
            "role": "system",
            "content": "You are a medical expert specializing in disease identification and treatment recommendations."
        },
        {
            "role": "user",
            "content": f"Here are the extracted clinical entities: {entities}. Based on these entities, please suggest possible diseases and treatment options."
        }
    ]
    data = {
        "model": "llama_405b",
        "messages": messages,
    }
    response = requests.post(llama_api_base, headers=headers, json=data)
    
    if response.status_code == 200:
        response_json = response.json()
        return response_json['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code} - {response.text}"

def format_disease_suggestions(response_text):
    formatted_text = response_text.replace("*", "-")
    
    formatted_text = re.sub(r'(\n\s*\n)', '\n\n', formatted_text)
    
    formatted_text = re.sub(r'(\*\*[^*]+\*\*)', r'**\1**', formatted_text)
    
    return formatted_text

def map_relationships_contextually(text, entities):
    relationship_mapping = []

    sentences = text.split('.')  
    for sentence in sentences:
        sentence_entities = extract_and_format_clinical_entities(sentence)[1]

        for diagnosis in sentence_entities.get("Diagnosis", []):
            related_anatomical_locations = sentence_entities.get("Anatomical Location", [])
            related_symptoms = sentence_entities.get("Symptom", [])
            
            if related_anatomical_locations or related_symptoms:
                relationship_mapping.append({
                    "Diagnosis": diagnosis,
                    "Anatomical Locations": ", ".join(related_anatomical_locations),
                    "Symptoms": ", ".join(related_symptoms)
                })
    
    return pd.DataFrame(relationship_mapping)

def highlight_entities(text, entities):
    color_mapping = {
        "Diagnosis": "lightblue",
        "Symptom": "lightgreen",
        "Anatomical Location": "lightcoral",
        "Treatment Recommendation": "lightgoldenrodyellow",
        "Lab Value": "lightpink",
        "Detailed Description": "lightgray",
        "Area": "lightcyan",
        "Other":"yellow"
    }

    for category, entity_list in entities.items():
        for entity in entity_list:
            pattern = re.compile(re.escape(entity), re.IGNORECASE)
            text = pattern.sub(f'<span style="background-color: {color_mapping.get(category, "lightyellow")};">{entity}</span>', text)

    return text

def color_rows_by_category(row):
    color_mapping = {
        "Diagnosis": "background-color: lightblue",
        "Symptom": "background-color: lightgreen",
        "Anatomical Location": "background-color: lightcoral",
        "Treatment Recommendation": "background-color: lightgoldenrodyellow",
        "Lab Value": "background-color: lightpink",
        "Detailed Description": "background-color: lightgray",
        "Area": "background-color: lightcyan",
        "Other":"background-color: yellow;"
    }
    return [color_mapping.get(row['Category'], '')] * len(row)

def download_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Function to download dataframe as JSON
def download_json(df):
    return df.to_json(orient='records', lines=True).encode('utf-8')

def main():
    uploaded_file = st.file_uploader("Upload a PDF, DOCX", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = extract_text_from_txt(uploaded_file)
        else:
            text = "Unsupported file type."

        if text:
            cleaned_text = preprocess_text(text)

            if "show_cleaned_text" not in st.session_state:
                st.session_state.show_cleaned_text = True
            if "show_entities" not in st.session_state:
                st.session_state.show_entities = False
            if "show_extracted_text_heading" not in st.session_state:
                st.session_state.show_extracted_text_heading = True
            if "show_download_options" not in st.session_state:
                st.session_state.show_download_options = False
            if "show_disease_detection" not in st.session_state:
                st.session_state.show_disease_detection = False

            if st.session_state.show_cleaned_text and st.session_state.show_extracted_text_heading:
                st.subheader("Original Text and Extracted Text Comparison")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Original Text")
                    st.markdown(cleaned_text, unsafe_allow_html=True)

                with col2:
                    detailed_entities, entities = extract_and_format_clinical_entities(cleaned_text)

                    if entities:
                        highlighted_text = highlight_entities(cleaned_text, entities)
                        st.subheader("Highlighted Extracted Text")
                        st.markdown(highlighted_text, unsafe_allow_html=True)

                if st.button("Extracted Entities"):
                    st.session_state.show_cleaned_text = True
                    st.session_state.show_entities = True
                    st.session_state.show_extracted_text_heading = True

            if st.session_state.show_entities:
                detailed_entities, entities = extract_and_format_clinical_entities(cleaned_text)

                if entities:
                    entity_df = pd.DataFrame(columns=["Category", "Entity", "Label", "Score"])
                    

                    for category, entity_list in entities.items():
                        for entity in entity_list:
                            matching_entity = next((e for e in detailed_entities if e["Entity"] == entity), None)
                            label = matching_entity["Label"] if matching_entity else "Unknown"
                            score = matching_entity["Score"] if matching_entity else "N/A"

                            entity_df = pd.concat([entity_df, pd.DataFrame({
                                "Category": [category], 
                                "Entity": [entity], 
                                "Label": [label], 
                                "Score": [score]
                            })])

                    entity_df = entity_df.reset_index(drop=True)

                    st.subheader("Extracted Clinical Entities")
                    st.dataframe(entity_df.style.apply(color_rows_by_category, axis=1), use_container_width=True)

                    summary_data = []
                    for category, entity_list in entities.items():
                        unique_entities = set(entity_list)
                        summary_data.append({
                            "Category": category,
                            "Entity Count": len(unique_entities),
                            "Entities": ", ".join(unique_entities)
                        })

                    summary_df = pd.DataFrame(summary_data)

                    st.subheader("Entity Summary View")
                    st.dataframe(summary_df.style.set_properties(
                        **{
                            'text-align': 'left',
                            'width': '10px'
                        }
                    ).set_table_styles(
                        [{'selector': 'td', 'props': 'padding: 5px;'},
                         {'selector': 'th', 'props': 'width: 80px;'}]
                    ), use_container_width=True)

                    st.write("Select download format:")
                    format_option = st.radio("Format", ["JSON", "CSV"])

                    if format_option == "CSV":
                        csv_data = download_csv(entity_df)
                        st.download_button(
                            label="Download as CSV",
                            data=csv_data,
                            file_name="extracted_entities.csv",
                            mime="text/csv"
                        )
                    elif format_option == "JSON":
                        json_data = download_json(entity_df)
                        st.download_button(
                            label="Download as JSON",
                            data=json_data,
                            file_name="extracted_entities.json",
                            mime="application/json"
                        )

                    if st.button("Disease Detection"):
                        st.session_state.show_disease_detection = True

            if st.session_state.show_disease_detection:
                disease_suggestions = analyze_clinical_entities_azure(entities)
                formatted_suggestions = format_disease_suggestions(disease_suggestions)

                st.subheader("Disease Identification and Treatment Recommendations")
                st.markdown(formatted_suggestions)

                st.subheader("Relationship Mapping")
                relationship_df = map_relationships_contextually(cleaned_text, entities)
                st.dataframe(relationship_df, use_container_width=True)

                st.session_state.show_disease_detection = False

if __name__ == "__main__":
    main()


footer_css = """
<style>
.footer {
    position: fixed;
    z-index: 1000;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
}
[data-testid="stSidebarNavItems"] {
    max-height: 100%!important;
}
[data-testid="collapsedControl"] {
    display: none;
}
</style>
"""

footer_html = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
<div style="text-align: center;">
    <p>
        Copyright © 2024 | <a href="https://trigent.com/ai/" target="_blank" aria-label="Trigent Website">Trigent Software Inc.</a> All rights reserved. |
        <a href="https://www.linkedin.com/company/trigent-software/" target="_blank" aria-label="Trigent LinkedIn"><i class="fab fa-linkedin"></i></a> |
        <a href="https://www.twitter.com/trigentsoftware/" target="_blank" aria-label="Trigent Twitter"><i class="fab fa-twitter"></i></a> |
        <a href="https://www.youtube.com/channel/UCNhAbLhnkeVvV6MBFUZ8hOw" target="_blank" aria-label="Trigent Youtube"><i class="fab fa-youtube"></i></a>
    </p>
</div>
"""

footer = f"{footer_css}<div class='footer'>{footer_html}</div>"

st.markdown(footer, unsafe_allow_html=True)