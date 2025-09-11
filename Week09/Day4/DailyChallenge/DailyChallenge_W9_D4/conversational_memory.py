"""
Getting Started with LangChain Conversational Memory

This script demonstrates how to use LangChain's ConversationBufferMemory
to store and retrieve conversation history for context-aware chatbots.
"""

# 1. Import the Memory Module
from langchain.memory import ConversationBufferMemory

def main():
    print("=== LangChain Conversational Memory Demo ===\n")
    
    # 2. Initialize the Memory
    print("1. Initializing ConversationBufferMemory...")
    memory = ConversationBufferMemory()
    print("✓ Memory initialized successfully\n")
    
    # 3. Simulate a First Interaction
    print("2. Saving first conversation exchange...")
    user_input_1 = "Hello, how are you?"
    bot_output_1 = "I'm fine, thank you. How can I assist you today?"
    
    # Save the first exchange using save_context
    memory.save_context(
        {"input": user_input_1}, 
        {"output": bot_output_1}
    )
    print(f"User: {user_input_1}")
    print(f"Bot: {bot_output_1}")
    print("✓ First exchange saved to memory\n")
    
    # 4. Simulate a Follow-up Message
    print("3. Saving second conversation exchange...")
    user_input_2 = "Tell me a joke."
    bot_output_2 = "Why did the chicken cross the road? To get to the other side."
    
    # Save the second exchange using save_context
    memory.save_context(
        {"input": user_input_2}, 
        {"output": bot_output_2}
    )
    print(f"User: {user_input_2}")
    print(f"Bot: {bot_output_2}")
    print("✓ Second exchange saved to memory\n")
    
    # 5. Retrieve the Conversation History
    print("4. Retrieving conversation history...")
    conversation_history = memory.load_memory_variables({})
    print("✓ Conversation history loaded successfully\n")
    
    # 6. Print the Memory (Optional)
    print("5. Full Conversation History:")
    print("=" * 50)
    print(conversation_history)
    print("=" * 50)
    
    # Additional demonstration: Show the formatted history
    print("\n6. Formatted Conversation History:")
    print("-" * 30)
    if 'history' in conversation_history:
        print(conversation_history['history'])
    
    # Show memory buffer contents in a more readable format
    print("\n7. Memory Buffer Details:")
    print("-" * 30)
    print(f"Memory type: {type(memory).__name__}")
    print(f"Number of messages stored: {len(memory.chat_memory.messages)}")
    
    # Display individual messages
    print("\nIndividual messages in memory:")
    for i, message in enumerate(memory.chat_memory.messages, 1):
        print(f"  {i}. {type(message).__name__}: {message.content}")

if __name__ == "__main__":
    main()