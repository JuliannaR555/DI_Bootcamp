import streamlit as st
import os
import csv
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

def get_vector_store(chunks):
    """Generate embeddings and create FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Build the conversational retrieval chain with custom prompt."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    Make sure to provide all the details. If the answer is not available in the context, 
    just say "answer is not available in the context". Don't provide the wrong answer.
    
    Context:
    {context}
    
    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    """Reset chat messages to default assistant prompt."""
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

def user_input(user_question):
    """Process user question and return AI response based on PDF content."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    context = "\n".join([doc.page_content for doc in docs])
    response = chain(
        {"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True, )

    return response['output_text']

def save_user_info(name, phone, email):
    """Save user contact information to CSV file."""
    file_exists = os.path.isfile('user_info.csv')
    with open('user_info.csv', mode='a', newline='') as file:
        fieldnames = ['Name', 'Phone', 'Email']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'Name': name, 'Phone': phone, 'Email': email})

def main():  
    st.set_page_config(  # Configure the Streamlit page settings
        page_title="Chat with PDF using Gemini",  # Set the browser tab title
        page_icon="🤖",  # Set the favicon/icon for the page
        layout="wide"  # Use a wide layout for more horizontal space
    ) 

    # Sidebar for uploading PDF files
    with st.sidebar:  # Begin sidebar container
        st.title("Menu")  # Display the title "Menu" in the sidebar
        pdf_docs = st.file_uploader(  # file uploader widget
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)  # Allow multiple PDF uploads
        if st.button("Submit & Process"):  # Add a button "Submit & Process" to trigger processing
            with st.spinner("Processing..."):  # Show a spinner while processing
                raw_text = get_pdf_text(pdf_docs)  # Extract raw text from uploaded PDFs
                text_chunks = get_text_chunks(raw_text)  # Split the text into smaller chunks
                get_vector_store(text_chunks)  # Build or update vector store for retrieval
                st.success("Done")  # Show a success message once processing finishes

    # Main content area for displaying chat messages
    st.title("Chat with PDF files using Gemini 🙋‍♂️")  # Display the main page title "Chat with PDF files using Gemini 🙋‍♂️"
    st.write("Welcome to the chat!")  # Display a welcome message "Welcome to the chat!"
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)  # Add a button 'Clear Chat History' in the sidebar to clear chat history with the function clear_chat_history

    # Chat input
    if "messages" not in st.session_state.keys():  # If no messages stored yet
        st.session_state.messages = [  # Initialize chat history with a default assistant message "Upload some PDFs and ask me a question"
            {"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

    for message in st.session_state.messages:  # Loop over stored chat messages
        with st.chat_message(message["role"]):  # Render each message with the correct role (user/assistant)
            st.markdown(f"**{message['role'].capitalize()}:** {message['content']}")  # Display the message content in markdown

    if prompt := st.chat_input():  # If the user enters a new chat prompt
        st.session_state.messages.append({"role": "user", "content": prompt})  # Add the user message to session state with st.session_state.messages.append
        with st.chat_message("user"):  # Render the user's message in the chat
            st.markdown(f"**User:** {prompt}")  # Display the user's input

        # Check for specific user request to call them
        if "call me" in prompt.lower():  # If prompt contains 'call me'
            st.session_state.collecting_info = True  # Flag to start collecting user contact info

        if st.session_state.messages[-1]["role"] != "assistant":  # If the last message isn't from the assistant yet
            with st.chat_message("assistant"):  # Prepare to render the assistant's response
                with st.spinner("Thinking..."):  # Show a spinner while the model generates a response
                    response = user_input(prompt)  # Call the retrieval+LLM pipeline to get an answer
                    st.session_state.messages.append({"role": "assistant", "content": response})  # Store the assistant's reply
                    st.markdown(f"**Assistant:** {response}")  # Display the assistant's response

    # Collect user information
    if "collecting_info" in st.session_state and st.session_state.collecting_info:  # If flagged to collect info
        st.subheader("Contact Information")  # Prompt for contact form with st.subheader
        with st.form(key="contact_form"):  # Begin a form for user contact details
            name = st.text_input("Name")  # Input field for name
            phone = st.text_input("Phone Number")  # Input field for phone number
            email = st.text_input("Email Address")  # Input field for email address
            submit_button = st.form_submit_button("Submit Contact Info")  # Button to submit the form with form_submit_button

            if submit_button:  # When the form is submitted
                save_user_info(name, phone, email)  # Save the user info to a CSV file
                st.session_state.messages.append({"role": "assistant", "content": f"Thank you, {name}. We will contact you at {phone} or {email}."})  # Thank the user
                st.session_state.collecting_info = False  # Stop collecting info

if __name__ == "__main__":  # Entry point check
    main()  # Run the main function
