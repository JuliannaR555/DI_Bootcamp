import streamlit as st
import os
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List
import time
from config import *


class StreamlitCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for streaming tokens to Streamlit."""
    
    def __init__(self, container):
        self.container = container
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Handle new token from LLM."""
        self.text += token
        self.container.markdown(self.text)


@st.cache_resource
def load_model():
    """Download and load the quantized Mistral model."""
    try:
        # Download the model from Hugging Face
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=MODEL_DIR
        )
        
        # Initialize the LlamaCpp model
        llm = LlamaCpp(
            model_path=model_path,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            verbose=True,
            streaming=True,
            n_ctx=CONTEXT_WINDOW,
            n_batch=BATCH_SIZE,
        )
        
        return llm
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def create_prompt_template():
    """Create the chat prompt template with system message and memory."""
    template = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("ai", "")
    ])
    return template


def initialize_memory():
    """Initialize conversation memory."""
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )


def create_chain(llm, memory, system_prompt):
    """Create the LangChain pipeline."""
    prompt_template = create_prompt_template()
    
    def format_chat_history(memory_dict):
        """Format chat history for the prompt."""
        return memory_dict.get("chat_history", [])
    
    def save_to_memory(inputs_and_output):
        """Save the conversation to memory."""
        inputs, output = inputs_and_output
        memory.save_context(
            {"input": inputs["input"]}, 
            {"output": output}
        )
        return output
    
    # Create the chain
    chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: format_chat_history(memory.load_memory_variables({})),
            system_prompt=lambda x: system_prompt
        )
        | prompt_template
        | llm
        | RunnableLambda(lambda output: save_to_memory(({"input": ""}, output)))
    )
    
    return chain


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide"
    )
    
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown("*Powered by Mistral 7B running locally with llama.cpp*")
    
    # Sidebar for system prompt configuration
    with st.sidebar:
        st.header("🎭 Configure Your Assistant")
        st.markdown("Define your AI assistant's personality and behavior:")
        
        system_prompt = st.text_area(
            "System Prompt",
            value=DEFAULT_SYSTEM_PROMPT,
            height=150,
            help="This defines how your AI assistant will behave and respond."
        )
        
        # Preset system prompts
        st.markdown("**Quick Presets:**")
        for preset_name, preset_prompt in PRESET_PROMPTS.items():
            if st.button(preset_name):
                st.session_state.system_prompt = preset_prompt
        
        # Reset conversation button
        if st.button("🔄 Clear Conversation"):
            if 'memory' in st.session_state:
                st.session_state.memory.clear()
            if 'messages' in st.session_state:
                st.session_state.messages = []
            st.rerun()
    
    # Update system prompt if changed
    if 'system_prompt' in st.session_state:
        system_prompt = st.session_state.system_prompt
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'memory' not in st.session_state:
        st.session_state.memory = initialize_memory()
    
    if 'llm' not in st.session_state:
        with st.spinner("Loading AI model... This may take a few minutes on first run."):
            st.session_state.llm = load_model()
    
    # Check if model loaded successfully
    if st.session_state.llm is None:
        st.error("Failed to load the AI model. Please check your internet connection and try again.")
        st.stop()
    
    # Create the chain
    chain = create_chain(st.session_state.llm, st.session_state.memory, system_prompt)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            message_container = st.empty()
            
            try:
                # Create streaming callback
                callback_handler = StreamlitCallbackHandler(message_container)
                
                # Configure the LLM with streaming callback
                st.session_state.llm.callbacks = [callback_handler]
                
                # Generate response
                with st.spinner("Thinking..."):
                    response = chain.invoke({"input": prompt})
                
                # Update memory with the new interaction
                st.session_state.memory.save_context(
                    {"input": prompt},
                    {"output": response}
                )
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                message_container.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Display some helpful information
    with st.expander("ℹ️ About this Chatbot"):
        st.markdown("""
        This chatbot uses:
        - **Mistral 7B Instruct** (quantized Q4_0 model) for natural language processing
        - **llama.cpp** for efficient local inference
        - **LangChain** for memory management and prompt engineering
        - **Streamlit** for the interactive user interface
        
        **Features:**
        - 🧠 **Memory**: Remembers previous messages in the conversation
        - 🎭 **Customizable Personality**: Modify the system prompt to change behavior
        - 🚀 **Local Processing**: Everything runs on your machine (no data sent to external servers)
        - ⚡ **Streaming**: Real-time response generation
        
        **Tips:**
        - Try different system prompts to change the assistant's personality
        - Ask follow-up questions to test the memory functionality
        - Use the preset buttons in the sidebar for quick personality changes
        """)


if __name__ == "__main__":
    main()