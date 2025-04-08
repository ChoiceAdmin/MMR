import os
import streamlit as st
from google import genai
from google.genai import types

# -----------------------
# Setup: Credentials
# -----------------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/sundeep.v/Downloads/interactive-ai-01-454009-bf54a9ac7759.json"

# -----------------------
# Helper Functions
# -----------------------

def generate_summary(pdf_bytes):
    """
    Generates a one-line summary of the PDF content using GenAI.
    """
    client = genai.Client(vertexai=True, project="interactive-ai-01-454009", location="us-central1")
    summary_prompt = "Please provide a one-line summary of the content of the following PDF."
    contents = [
         types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=summary_prompt),
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            ]
         )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.8,
        max_output_tokens=256,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ],
    )
    response_text = ""
    for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash-001",
            contents=contents,
            config=generate_content_config,
    ):
        response_text += chunk.text
    return response_text.strip()

def init_conversation(pdf_bytes):
    """
    Initializes the conversation context using a prompt and including the PDF bytes.
    """
    initial_prompt = (
        "You will be provided with a PDF file. Please answer all user queries based only on the "
        "information contained within the PDF file. If a query cannot be answered based on the PDF, "
        "respond with: 'I cannot answer this question based on the information in the provided PDF.'"
    )
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=initial_prompt),
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            ]
        )
    ]
    return contents

def generate_chat_response(contents):
    """
    Given a conversation context (contents), generate a response using GenAI.
    """
    client = genai.Client(vertexai=True, project="interactive-ai-01-454009", location="us-central1")
    generate_content_config = types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.8,
        max_output_tokens=2048,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ],
    )
    response_text = ""
    for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash-001",
            contents=contents,
            config=generate_content_config,
    ):
        response_text += chunk.text
    return response_text.strip()

# -----------------------
# Streamlit Page Configuration & Custom CSS
# -----------------------
st.set_page_config(page_title="MMR - Multi Model RAG", layout="wide")
st.markdown("""
    <style>
    body { background-color: #F4F6F8; }
    .main > div { max-width: 1200px; margin: 0 auto; padding-top: 2rem; }
    h1 { 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
        color: #072A40; 
        text-align: center; 
        margin-bottom: 2rem; 
    }
    .stCard { 
        background-color: #fff; 
        border-radius: 8px; 
        padding: 1.5rem; 
        box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        margin-bottom: 1rem; 
    }
    .stCard > h3 { 
        margin-top: 0; 
        margin-bottom: 1rem; 
        color: #072A40; 
    }
    .stButton button { 
        background-color: #072A40 !important; 
        color: #fff !important; 
        border: none !important; 
        border-radius: 5px !important; 
        font-weight: 500 !important; 
        padding: 0.6rem 1.2rem !important; 
        margin-top: 0.5rem !important; 
    }
    .stButton button:hover { 
        background-color: #09486A !important; 
    }
    .uploadedFileName { 
        font-size: 0.9rem; 
        color: #555; 
        margin-top: 0.5rem; 
    }
    input[type="text"] { 
        border-radius: 5px !important; 
        padding: 0.5rem !important; 
        color: #072A40 !important; 
    }
    textarea { 
        width: 100%; 
        height: 200px; 
        border: 1px solid #ccc; 
        border-radius: 5px; 
        padding: 0.5rem; 
        resize: vertical; 
        color: #072A40 !important;
    }
    ::placeholder { 
        color: #555 !important; 
    }
    </style>
""", unsafe_allow_html=True)

st.title("MMR - Multi Model RAG")

# -----------------------
# Layout: Two Columns (Left: File Upload, Right: Chat Bot)
# -----------------------
col_left, col_right = st.columns(2, gap="large")

# --- LEFT COLUMN: File Upload and Report Summary ---
with col_left:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.header("Upload Documents")
    
    uploaded_file = st.file_uploader("Choose File", type=["pdf", "txt", "docx"], label_visibility="collapsed")
    
    if uploaded_file:
        pdf_bytes = uploaded_file.read()
        st.session_state["pdf_bytes"] = pdf_bytes

        # Generate summary if not already in session state.
        if "summary" not in st.session_state:
            with st.spinner("Generating summary..."):
                summary = generate_summary(pdf_bytes)
            st.session_state["summary"] = summary
        else:
            summary = st.session_state["summary"]

        # Initialize conversation context if not already done.
        if "contents" not in st.session_state:
            st.session_state["contents"] = init_conversation(pdf_bytes)

        st.success("PDF uploaded successfully!")
        st.markdown("**Sample Report Summary:**")
        st.write(summary)
    else:
        st.info("Please upload a PDF file to initialize the conversation.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- RIGHT COLUMN: Chat Bot Interface ---
with col_right:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.header("Chat Bot")
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    # Display chat history.
    chat_display = ""
    for message in st.session_state["chat_history"]:
        if message["role"] == "user":
            chat_display += f"**You:** {message['content']}\n\n"
        else:
            chat_display += f"**Bot:** {message['content']}\n\n"
    st.text_area("Chat History", value=chat_display, height=200, disabled=True)
    
    # User input for questions.
    user_query = st.text_input("Ask a question...", key="chat_input")
    if st.button("Send"):
        if not user_query:
            st.warning("Please enter a query.")
        elif "contents" not in st.session_state:
            st.error("Please upload a PDF file first.")
        else:
            # Append the user query to both chat history and conversation context.
            st.session_state["chat_history"].append({"role": "user", "content": user_query})
            st.session_state["contents"].append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_query)]
                )
            )
            with st.spinner("Generating response..."):
                bot_response = generate_chat_response(st.session_state["contents"])
            st.session_state["chat_history"].append({"role": "bot", "content": bot_response})
            st.session_state["contents"].append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=bot_response)]
                )
            )
            st.experimental_rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)
