from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import torch
from api.chroma_utils import vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

output_parser = StrOutputParser()

# Set up prompts and chains
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


qa_system_prompt = (
    "You are a helpful AI assistant. Use the following context to answer the "
    "user's question. If the context is empty or not relevant, feel free to "
    "answer the question conversationally based on your own knowledge.\n\n"
    "Context: {context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])




# Load OuteAI/Lite-Oute-1-300M-Instruct model and tokenizer
_hf_model_name = "OuteAI/Lite-Oute-1-300M-Instruct"
_tokenizer = AutoTokenizer.from_pretrained(_hf_model_name)
_model = AutoModelForCausalLM.from_pretrained(_hf_model_name)

def local_llm_chat(prompt, history=None, max_new_tokens=512):
    # Optionally use history for context
    input_text = prompt
    if history:
        input_text = "\n".join(history + [prompt])
    # Limit input_text length (e.g., 2048 chars)
    max_context_chars = 2048
    if len(input_text) > max_context_chars:
        input_text = input_text[-max_context_chars:]
    inputs = _tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
    response = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove prompt from response if model echoes it
    if response.startswith(input_text):
        response = response[len(input_text):].strip()
    return response

def get_rag_chain(model=None):
    # Use local LLM and retriever
    class LocalRAGChain:
        def __init__(self, retriever):
            self.retriever = retriever
        def invoke(self, inputs):
            question = inputs["input"]
            chat_history = [msg.content for msg in inputs.get("chat_history", [])]
            # Retrieve context
            docs = self.retriever.get_relevant_documents(question)
            # Limit context to first 1200 characters (or fewer docs)
            context_chunks = [doc.page_content for doc in docs]
            context = "\n".join(context_chunks)
            max_context_chars = 1200
            if len(context) > max_context_chars:
                context = context[:max_context_chars]
            # Compose prompt
            prompt = f"You are a helpful AI assistant. Use the following context to answer the user's question. If the context is empty or not relevant, answer conversationally.\n\nContext: {context}\n\nQuestion: {question}"
            answer = local_llm_chat(prompt, history=chat_history, max_new_tokens=512)
            return {"answer": answer}
    return LocalRAGChain(retriever)