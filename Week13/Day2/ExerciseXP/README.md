# Antarctic Agents: Multi-Agent Simulation

This project implements a multi-agent simulation with penguins and a scientist using smolagents and Hugging Face models.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Hugging Face token:**
   - Get a free token from https://huggingface.co/settings/tokens
   - Copy `.env.example` to `.env`
   - Set your `HUGGINGFACEHUB_API_TOKEN` in the `.env` file

3. **Run the simulation:**
   ```bash
   python exercises/starter.py
   ```

## What the simulation does

- **Penguin Agents**: Use the `find_food` tool to fish or forage for food
- **Scientist Agent**: Monitors penguin food levels and distributes supplies when needed
- **Multi-round simulation**: Runs for 3 rounds with decision-making and interactions

## Key Features

✅ **Tool Implementation**: `find_food` tool with fishing (2-7 food) vs foraging (0-3 food)  
✅ **Agent Registration**: Penguins use the tool via smolagents  
✅ **JSON Actions**: Structured decision-making with JSON output  
✅ **Multi-agent Communication**: Scientist responds to penguin needs  
✅ **State Tracking**: Food levels and distribution history  

## Expected Output

The simulation shows:
- Penguin actions and food gathering
- Scientist supply distributions
- State changes across 3 rounds
- Final food totals and distribution history

Enjoy watching the Antarctic ecosystem in action! 🐧🔬
