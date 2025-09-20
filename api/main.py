from fastapi import FastAPI, File, UploadFile, HTTPException
import uuid
import logging

from api.pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from api.langchain_utils import get_rag_chain
from api.db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from api.chroma_utils import index_document_to_chroma, delete_doc_from_chroma

logging.basicConfig(filename='app.log', level=logging.INFO)
app = FastAPI()

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")
    if not session_id:
        session_id = str(uuid.uuid4())

    

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query_input.model.value)
    
    try:
        logging.info(f"Session ID: {session_id}, Chat History: {chat_history}")
        logging.info(f"Session ID: {session_id}, User Question: {query_input.question}")

        result = rag_chain.invoke({
            "input": query_input.question,
            "chat_history": chat_history
        })
        answer = result['answer']

        insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    except Exception as e:
        logging.error(f"Session ID: {session_id}, Error during RAG chain invocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)

from fastapi import UploadFile, File, HTTPException
import os
import shutil

@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()
    logging.info(f"Received file upload: {file.filename} (ext: {file_extension})")
    if file_extension not in allowed_extensions:
        logging.error(f"Unsupported file type: {file.filename}")
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")
    temp_file_path = f"temp_{file.filename}"
    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Saved temp file: {temp_file_path}")
        file_id = insert_document_record(file.filename)
        logging.info(f"Inserted document record in DB with file_id: {file_id}")
        success = index_document_to_chroma(temp_file_path, file_id)
        logging.info(f"Indexing to ChromaDB returned: {success}")
        if success:
            logging.info(f"File {file.filename} successfully indexed with file_id {file_id}")
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            logging.error(f"Failed to index {file.filename}, deleting DB record.")
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logging.info(f"Removed temp file: {temp_file_path}")

@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()

@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    # Delete from Chroma
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        # If successfully deleted from Chroma, delete from our database
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            # This is an inconsistent state. Log it and raise an error.
            logging.error(f"CRITICAL: Deleted from Chroma but failed to delete from DB. file_id: {request.file_id}")
            raise HTTPException(status_code=500, detail=f"Critical: Document deleted from vector store but failed to delete from database (ID: {request.file_id}). Please check server logs.")
    else:
        raise HTTPException(status_code=500, detail=f"Failed to delete document with file_id {request.file_id} from the vector store.")
