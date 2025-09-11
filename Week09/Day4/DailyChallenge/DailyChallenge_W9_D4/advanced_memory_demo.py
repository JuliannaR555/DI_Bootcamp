"""
Advanced LangChain Conversational Memory Example

This script demonstrates how to integrate ConversationBufferMemory 
with a simple chatbot simulation to show how memory enhances conversations.
"""

from langchain.memory import ConversationBufferMemory

class SimpleChatbot:
    """A simple chatbot that uses ConversationBufferMemory to maintain context."""
    
    def __init__(self):
        # Initialize memory to store conversation history
        self.memory = ConversationBufferMemory()
        self.name = "Assistant"
        
    def get_response(self, user_input):
        """
        Generate a response based on user input and conversation history.
        This is a simplified version - in a real implementation, 
        you'd use an LLM like OpenAI's GPT or similar.
        """
        # Load previous conversation context
        context = self.memory.load_memory_variables({})
        history = context.get('history', '')
        
        # Simple rule-based responses (in reality, you'd use an LLM here)
        user_input_lower = user_input.lower()
        
        if "hello" in user_input_lower or "hi" in user_input_lower:
            if history:  # If there's previous context
                response = "Hello again! How can I continue to help you?"
            else:
                response = "Hello! I'm your AI assistant. How can I help you today?"
                
        elif "joke" in user_input_lower:
            if "chicken" in history:  # Memory helps avoid repeating jokes
                response = "Here's a different one: Why don't scientists trust atoms? Because they make up everything!"
            else:
                response = "Why did the chicken cross the road? To get to the other side."
                
        elif "name" in user_input_lower:
            if "assistant" in history.lower():  # Memory shows we already introduced ourselves
                response = "As I mentioned before, I'm your AI Assistant!"
            else:
                response = "I'm your AI Assistant. Nice to meet you!"
                
        elif "remember" in user_input_lower or "said" in user_input_lower:
            if history:
                response = f"Yes, I remember our conversation. Here's what we discussed:\n{history}"
            else:
                response = "This is the start of our conversation, so there's nothing to remember yet."
                
        else:
            response = "I understand. Is there anything specific I can help you with?"
        
        # Save this exchange to memory
        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )
        
        return response
    
    def get_conversation_summary(self):
        """Get a summary of the entire conversation."""
        context = self.memory.load_memory_variables({})
        return context.get('history', 'No conversation history yet.')
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()

def demo_conversation():
    """Demonstrate a conversation with memory."""
    print("=== Advanced Conversational Memory Demo ===\n")
    
    # Create chatbot instance
    bot = SimpleChatbot()
    
    # Simulate a conversation
    conversations = [
        "Hello there!",
        "What's your name?",
        "Tell me a joke",
        "Do you remember what I first said to you?",
        "Tell me another joke",
        "What have we talked about so far?"
    ]
    
    print("Starting conversation simulation...\n")
    
    for i, user_message in enumerate(conversations, 1):
        print(f"Turn {i}:")
        print(f"User: {user_message}")
        
        response = bot.get_response(user_message)
        print(f"Bot: {response}")
        print("-" * 50)
    
    # Show final conversation summary
    print("\nFinal Conversation Summary:")
    print("=" * 60)
    summary = bot.get_conversation_summary()
    print(summary)
    print("=" * 60)
    
    # Show memory statistics
    memory_vars = bot.memory.load_memory_variables({})
    message_count = len(bot.memory.chat_memory.messages)
    print(f"\nMemory Statistics:")
    print(f"Total messages stored: {message_count}")
    print(f"Memory type: {type(bot.memory).__name__}")

if __name__ == "__main__":
    demo_conversation()