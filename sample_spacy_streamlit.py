import spacy
import spacy_streamlit
import streamlit as st

# List of SpaCy models to choose from
models = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]

# Streamlit interface for selecting a model
selected_model = st.selectbox("Select SpaCy Model:", models)

# Load the selected model
@st.cache_resource
def load_model(model_name):
    return spacy.load(model_name)

nlp = load_model(selected_model)

# Streamlit interface for text input
st.title("Named Entity Recognition with spaCy Streamlit")
text = st.text_area("Enter text for NER:")

if text:
    doc = nlp(text)
    # Visualize NER with spacy-streamlit
    spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels)
