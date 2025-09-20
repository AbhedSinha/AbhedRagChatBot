import streamlit as st
from chat_interface import display_chat_interface
from ui_sidebar import display_sidebar

# --- Page Configuration ---
st.set_page_config(
    page_title="Abhed's RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

def inject_custom_css():
    """Injects custom CSS to apply a glassmorphism effect."""
    st.markdown("""
        <style>
            /* Add a background image to the main content area */
            [data-testid="stAppViewContainer"] > .main {
                background-image: url("https://images.unsplash.com/photo-1501854140801-50d01698950b?q=80&w=2400&auto=format&fit=crop");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }

            /* Make title and caption stand out on the background */
            h1 {
                color: white !important;
                text-shadow: 2px 2px 8px rgba(0,0,0,0.7), 0 0 20px rgba(0, 150, 255, 0.6), 0 0 35px rgba(255, 0, 150, 0.5), 0 0 50px rgba(0, 255, 255, 0.4);
            }

            [data-testid="stCaptionContainer"] {
                color: white !important;
                text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
            }

            /* Glassmorphism for the sidebar */
            [data-testid="stSidebar"] > div:first-child {
                background: rgba(20, 20, 30, 0.7);
                backdrop-filter: blur(12px) saturate(180%);
                -webkit-backdrop-filter: blur(12px) saturate(180%); /* For Safari */
                border-right: 1px solid rgba(255, 255, 255, 0.1);
            }

            /* Glassmorphism for the chat input bar with glowing effect */
            [data-testid="stChatInput"] {
                background: rgba(20, 20, 30, 0.7);
                backdrop-filter: blur(12px) saturate(180%);
                -webkit-backdrop-filter: blur(12px) saturate(180%);
                border-top: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1), 0 0 20px rgba(0, 150, 255, 0.5), 0 0 35px rgba(255, 0, 150, 0.4); /* Multicolor Glowing effect */
            }

            /* Glassmorphism for chat messages */
            [data-testid="stChatMessage"] {
                background: rgba(40, 50, 70, 0.7);
                backdrop-filter: blur(8px) saturate(150%);
                -webkit-backdrop-filter: blur(8px) saturate(150%);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 0.5rem;
            }

            /* Improve text visibility on glass background */
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
                color: white !important;
            }
            [data-testid="stChatInput"] textarea {
                color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

st.title("🤖 Abhed's Langchain RAG Chatbot")
st.caption("Powered by LangChain, Gemini, and ChromaDB. Chat with your documents!")

# --- Session State Initialization ---
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "model" not in st.session_state:
        st.session_state.model = "gemini-1.5-flash-latest"

initialize_session_state()

# --- Main Layout ---
display_sidebar()
display_chat_interface()