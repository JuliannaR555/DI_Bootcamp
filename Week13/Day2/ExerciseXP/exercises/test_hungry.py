"""
Test script to demonstrate the system with hungry penguins
"""

import os
import json
import random
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Set up to use a mock model that doesn't require HF token for testing
class MockClient:
    def __init__(self, model, token=None):
        self.model = model
        self.token = token

# Load the main simulation components
import sys
sys.path.append(os.path.dirname(__file__))

from starter import find_food, PenguinAgent, ScientistAgent

def test_hungry_scenario():
    """Test the simulation with hungry penguins to see scientist intervention."""
    print("🧪 Testing hungry penguin scenario...\n")
    
    # Create mock client
    client = MockClient("test-model")
    
    # Create agents with low initial food
    penguins = [
        PenguinAgent("Hungry", client),
        PenguinAgent("Starving", client),
    ]
    
    # Set them to be hungry
    penguins[0].food = 2  # Hungry
    penguins[1].food = 1  # Very hungry
    
    scientist = ScientistAgent(client)
    
    print("Initial state (hungry scenario):")
    for penguin in penguins:
        print(f"🐧 {penguin.name}: {penguin.food} food")
    print(f"🔬 Scientist: {scientist.supplies} supplies\n")
    
    # Run one round to see scientist intervention
    print("=== TESTING SCIENTIST INTERVENTION ===")
    
    # Scientist makes distributions for hungry penguins
    distributions = scientist.decide_distribution(penguins)
    
    # Apply distributions
    for penguin in penguins:
        if penguin.name in distributions:
            amount = distributions[penguin.name]
            penguin.food += amount
            print(f"🔬 Gave {amount} food to {penguin.name} (new total: {penguin.food})")
    
    print(f"\nAfter scientist intervention:")
    for penguin in penguins:
        print(f"🐧 {penguin.name}: {penguin.food} food")
    print(f"🔬 Scientist: {scientist.supplies} supplies")
    print(f"🔬 Distribution history: {scientist.distribution_history}")

if __name__ == "__main__":
    test_hungry_scenario()
