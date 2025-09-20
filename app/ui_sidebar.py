import streamlit as st
from api_utils import upload_document, list_documents, delete_document

def display_sidebar():
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        model_options = ["gemini-1.5-flash-latest"]
        st.selectbox("Select Model", options=model_options, key="model")

        st.divider()

        st.header("📚 Knowledge Base")

        # --- Upload Section ---
        with st.expander("Upload a Document"):
            uploaded_file = st.file_uploader(
                "Upload a .pdf, .docx, or .html file",
                type=["pdf", "docx", "html"],
                label_visibility="collapsed"
            )
            if uploaded_file:
                if st.button(f"Upload '{uploaded_file.name}'"):
                    with st.spinner("Processing..."):
                        upload_response = upload_document(uploaded_file)
                        if upload_response:
                            st.success("File uploaded successfully!")
                            st.session_state.documents = list_documents()
                            st.rerun()

        # --- Document List and Deletion Section ---
        st.subheader("Your Documents")

        if st.button("Refresh List 🔄"):
            with st.spinner("Refreshing..."):
                st.session_state.documents = list_documents()

        if "documents" not in st.session_state:
            st.session_state.documents = list_documents()

        documents = st.session_state.get("documents", [])

        if documents:
            doc_map = {doc['id']: doc['filename'] for doc in documents}
            selected_id = st.selectbox("Select a document to delete", options=list(doc_map.keys()), format_func=lambda doc_id: doc_map.get(doc_id, "Unknown"))
            if st.button("Delete Selected Document", type="primary"):
                with st.spinner(f"Deleting '{doc_map[selected_id]}'..."):
                    delete_response = delete_document(selected_id)
                    if delete_response:
                        st.success("Document deleted.")
                        st.session_state.documents = list_documents()
                        st.rerun()
        else:
            st.info("No documents found. Upload a file to begin.")