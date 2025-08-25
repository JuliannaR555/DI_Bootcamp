"""
Antarctic Agents: Multi-Agent Architecture Implementation
A simulation with penguin agents and a scientist using smolagents.
"""

import os
import json
import random
from typing import Dict, List, Any
from dotenv import load_dotenv
from smolagents import CodeAgent, tool
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "HuggingFaceH4/zephyr-7b-beta")

if not HF_TOKEN:
    raise ValueError("Please set HUGGINGFACEHUB_API_TOKEN in your .env file")


@tool
def find_food(penguin_name: str, method: str) -> int:
    """Return a small random food yield based on the penguin's foraging method.
    
    Args:
        penguin_name: The name of the penguin searching for food
        method: The foraging method - "fishing" returns 2-7 food, otherwise 0-3
    
    Returns:
        Amount of food found as an integer
    """
    if method == "fishing":
        food_yield = random.randint(2, 7)
        print(f"🐧 {penguin_name} went fishing and caught {food_yield} fish!")
    else:
        food_yield = random.randint(0, 3)
        print(f"🐧 {penguin_name} went foraging and found {food_yield} food items.")
    
    return food_yield


class PenguinAgent:
    """A penguin agent that can find food and request supplies."""
    
    def __init__(self, name: str, model):
        self.name = name
        self.food = random.randint(3, 8)  # Starting food
        self.agent = CodeAgent(tools=[find_food], model=model, name=name)
    
    def take_action(self) -> Dict[str, Any]:
        """Decide what action to take based on current state."""
        # Simple decision logic
        if self.food < 3:
            # Very low food - might request help
            if random.random() < 0.4:  # 40% chance to request food
                action = {"action": "request_food", "reason": "low_food"}
                print(f"🐧 {self.name} is requesting food (current: {self.food})")
                return action
        
        # Normal food finding behavior
        # Prefer fishing if available, otherwise foraging
        if hasattr(self.agent, 'tools') and find_food in self.agent.tools:
            method = "fishing" if random.random() > 0.3 else "foraging"
            action = {"action": "find_food", "method": method}
            print(f"🐧 {self.name} decided to use tool: {method}")
        else:
            action = {"action": "find_food", "method": "foraging"}
            print(f"🐧 {self.name} decided to forage")
        
        return action
    
    def execute_action(self, action: Dict[str, Any]) -> int:
        """Execute the decided action and return food gained."""
        if action["action"] == "find_food":
            # Use the tool to find food
            food_gained = find_food(self.name, action.get("method", "foraging"))
            self.food += food_gained
            return food_gained
        elif action["action"] == "request_food":
            print(f"🐧 {self.name} is waiting for scientist response...")
            return 0
        else:
            print(f"🐧 {self.name} took an unknown action")
            return 0


class ScientistAgent:
    """A scientist agent that monitors penguins and provides supplies."""
    
    def __init__(self, model):
        self.model = model
        self.supplies = 50
        self.distribution_history = []
    
    def check_history(self) -> str:
        """Check distribution history for context."""
        if not self.distribution_history:
            return "No previous distributions recorded."
        
        recent = self.distribution_history[-3:]  # Last 3 distributions
        summary = f"Recent distributions: {', '.join(map(str, recent))}"
        return summary
    
    def record_distribution(self, amount: int):
        """Record a distribution in history."""
        self.distribution_history.append(amount)
    
    def decide_distribution(self, penguins: List[PenguinAgent]) -> Dict[str, int]:
        """Decide how much food to give each penguin."""
        distributions = {}
        
        # Check who needs food most
        hungry_penguins = [p for p in penguins if p.food < 5]
        
        if not hungry_penguins:
            print("🔬 Scientist: All penguins seem well-fed.")
            return distributions
        
        # Simple distribution logic
        total_to_distribute = min(self.supplies, len(hungry_penguins) * 4)
        
        if total_to_distribute > 0:
            per_penguin = total_to_distribute // len(hungry_penguins)
            
            for penguin in hungry_penguins:
                if per_penguin > 0:
                    distributions[penguin.name] = per_penguin
                    self.supplies -= per_penguin
                    self.record_distribution(per_penguin)
            
            print(f"🔬 Scientist distributed {total_to_distribute} food to {len(hungry_penguins)} penguins")
            print(f"🔬 Remaining supplies: {self.supplies}")
        
        # Occasionally refresh supplies
        if random.random() < 0.3:  # 30% chance
            refresh_amount = random.randint(10, 20)
            self.supplies += refresh_amount
            print(f"🔬 Scientist received {refresh_amount} new supplies! Total: {self.supplies}")
        
        return distributions


def run_simulation():
    """Run the multi-agent simulation."""
    print("🐧 Starting Antarctic Agents Simulation 🔬\n")
    
    # Initialize model
    client = InferenceClient(model=HF_MODEL_ID, token=HF_TOKEN)
    
    # Create agents
    penguins = [
        PenguinAgent("Pip", client),
        PenguinAgent("Skipper", client),
        PenguinAgent("Waddle", client)
    ]
    
    scientist = ScientistAgent(client)
    
    print("Initial state:")
    for penguin in penguins:
        print(f"🐧 {penguin.name}: {penguin.food} food")
    print(f"🔬 Scientist: {scientist.supplies} supplies\n")
    
    # Run simulation for 3 rounds
    for round_num in range(1, 4):
        print(f"=== ROUND {round_num} ===")
        
        # Penguins take actions
        for penguin in penguins:
            print(f"\n--- {penguin.name}'s turn ---")
            action = penguin.take_action()
            food_gained = penguin.execute_action(action)
            
            if food_gained > 0:
                print(f"🐧 {penguin.name} gained {food_gained} food (total: {penguin.food})")
        
        # Scientist makes distributions
        print(f"\n--- Scientist's turn ---")
        history_context = scientist.check_history()
        print(f"🔬 {history_context}")
        
        distributions = scientist.decide_distribution(penguins)
        
        # Apply distributions
        for penguin in penguins:
            if penguin.name in distributions:
                amount = distributions[penguin.name]
                penguin.food += amount
                print(f"🔬 Gave {amount} food to {penguin.name} (new total: {penguin.food})")
        
        # Round summary
        print(f"\nRound {round_num} summary:")
        for penguin in penguins:
            print(f"🐧 {penguin.name}: {penguin.food} food")
        print(f"🔬 Scientist: {scientist.supplies} supplies")
        print()
    
    # Final results
    print("=== FINAL RESULTS ===")
    total_food = sum(p.food for p in penguins)
    print(f"Total penguin food: {total_food}")
    print(f"Scientist supplies remaining: {scientist.supplies}")
    print(f"Distribution history: {scientist.distribution_history}")
    
    print("\n🎉 Simulation completed successfully!")


if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        print("Make sure you have set up your .env file with a valid HUGGINGFACEHUB_API_TOKEN")
