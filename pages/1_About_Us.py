import streamlit as st

st.set_page_config(page_title="About Us", layout="centered")

st.title("About Us")

st.header("Project Scope")
st.markdown("""
PUB’s Code of Practice is an important reference document for engineering companies 
to build compliant and safe utility infrastructure. It consists of a few lengthy 
documents that can be difficult to navigate and read as there are frequent references 
to appendixes or drawings. The main project scope will be to build a question and answer bot 
that can provide answers by referencing the relevant parts of the lengthy documents.
""")

st.header("Objectives")
st.markdown("""
- **Use Case 1**: Build a retrieval and answering system for the private and public 
  sector to ask questions and receive required instructions and specifications without 
  the inconvenience of referring to the document.  
- **Use Case 2**: An upload-and-answer feature to allow users to upload their own set 
  of documents for the system to help answer questions.
""")

st.header("Data Sources")
st.markdown("""
- Code of Practice on Sewerage and Sanitary Works  
- Code of Practice on Surface Water Drainage
""")

st.header("Features")
st.markdown("""
- Upload and ask  
- Referencing drawings and appendix combined with main body of text
""")