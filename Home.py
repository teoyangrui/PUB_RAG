import atexit
import streamlit as st
from helper_functions import llm, helper  # modules exported by __init__.py

#configurations
VALID_USERNAME = "test_user"
VALID_PASSWORD = "test_user123!"

def show_login() -> bool:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if st.session_state.logged_in:
        return True

    st.title("🔐 Login")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Invalid username or password.")
    return False
st.markdown('''
IMPORTANT NOTICE: This web application is developed as a proof-of-concept prototype. The information provided here is NOT intended for actual usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.
Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full 

''')


with st.expander("⚠️ Disclaimer"):
    st.markdown(
        """
        This project is intended as a proof-of-concept (POC) prototype to ascertain the feasibility 
        and effectiveness of the proposed methodology.  
        The resulting web application is a proof-of-concept prototype and should be treated as such.  

        **Fair use includes:**  
        - Conducting user studies to collect more requirements and refine the prototype  
        - Demonstrating to stakeholders the potential of the methodology  
        - For sharing or discussion on the technical aspects of the application  

        **It MUST NOT be used by end users for actual use cases** and should not be relied upon for making 
        any decisions, especially those related to financial, legal, or healthcare matters.  

        **You MUST NOT:**  
        - Promote or distribute this application to the general public or a wider audience  
        - Use data that does not match the classification and sensitivity levels required by the tools, platforms, or API services you are utilizing  
        - Use this application for any purpose that could potentially harm or mislead users
        """,
        unsafe_allow_html=False,
    )
#App config
st.set_page_config(layout="centered", page_title="My Streamlit App")

#for uploaded documents, a temp vector db will be created to process the documents and then deleted at end of session
def _cleanup_on_exit():
    try:
        helper.clear_temp_chroma()
    except Exception:
        pass
atexit.register(_cleanup_on_exit)

#only allow access if login successful
if not show_login():
    st.stop()

#side bar for document uploads and login log out
with st.sidebar:
    st.caption(f"Signed in as **{st.session_state.username}**")

    st.header("📄 Optional uploaded context")
    uploaded = st.file_uploader(
        "Upload PDF/DOCX/TXT (session-only; not persisted)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    use_uploaded_context = st.checkbox("Use uploaded docs for this question", value=False)
    k = st.slider("Top results from uploaded docs", 3, 20, 8)
    strict = st.checkbox("Answer ONLY from uploaded docs", value=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear uploaded context"):
            helper.clear_temp_chroma()
            st.success("Temporary context cleared.")
    with col_b:
        if st.button("Log out"):
            helper.clear_temp_chroma()  # clear the temp store on logout
            for key in ("logged_in", "username"):
                st.session_state.pop(key, None)
            st.rerun()

#main page for user to ask questions
st.title("Streamlit App")
st.markdown('''
Example Questions to test:
1. What is the recommended cleaning procedures for a standard circular grease trap according to PUB?

2. What is the required minimum setback for a ≤600 mm sewer laid at >5 m depth, and how is the setback measured relative to structures? 
 

3. State the Public Sewer Corridor distances (X) on either side of the centreline for DTSS tunnels, sewers ≥900 mm, and sewers <900 mm. 

4. For a mixed development/commercial building, how must the drain-line connection to the public sewer be made, and in what cases is a ‘Y’-junction permitted? 

5. What are the minimum sewer size and the design/allowable peak-flow velocities (target, minimum, maximum) for newly constructed sewers? 
 

6. Before commissioning, what testing/inspection must be completed for new sewers/manholes/pumping mains/chambers, and is this repeated before the end of the DLP? 

7. For pumped drainage systems, what are the minimum standby pumping configurations required for General Developments vs other development types? (Answer in N + … form.) 

8. Define freeboard and state the general freeboard requirement as a percentage of drain depth at design flow. 

9. Upon TOP, who must make annual declarations for developments with flood protection measures, and what systems/measures do these declarations cover? 
 

10. For construction/earthwork sites, what water-quality limit must discharges meet under the regulations, and to what storm return period must ECM be designed? 
 

11. When an at-grade structure is built over a drain/drainage reserve, what maintenance openings and lay-bys must be provided to ensure access? (Give dimensions/arrangement.          

''')
form = st.form(key="form")
form.subheader("Prompt")
user_prompt = form.text_area("Enter your prompt here", height=200)
submitted = form.form_submit_button("Submit")

if submitted:
    if not user_prompt.strip():
        st.warning("Ask a question on PUB Code of Practice, or upload your own document.")
    else:
        st.toast(f"User Input Submitted - {user_prompt}")

        if use_uploaded_context and uploaded:
            # 1) Parse → chunk → segments
            segments = helper.build_segments_from_uploads(uploaded) # processing of uploaded doc to create temp vectorDB
            if not segments:
                st.info("No readable content found in the uploaded files. Proceeding with direct chat.")
                response = llm.ask(user_prompt)
            else:
                # 2) Index into temp Chroma
                helper.chroma_add_segments(segments)
                # 3) Retrieve top-k
                excerpts = helper.chroma_query(user_prompt, top_k=k)
                # 4) Ask with uploaded context retrieved from vector store
                response = helper.ask_with_temp_context(user_prompt, excerpts, strict=strict)

                with st.expander("🔎 Context used (from uploaded docs)"):
                    for i, e in enumerate(excerpts, 1):
                        src = e.get("metadata", {}).get("source", "uploaded")
                        page = e.get("metadata", {}).get("page", "?")
                        st.markdown(f"**{i}. {src} — p.{page}**")
                        st.write(e["text"])
        else:
            # The default - using vector store
            if use_uploaded_context and not uploaded:
                st.info("‘Use uploaded docs’ is on, but no files were uploaded. Answering from the main knowledge base.")
            response = llm.ask(user_prompt)

        st.markdown("### Answer")
        st.write(response)
        print(f"User Input is {user_prompt}")