import os
import re
import datetime
import time
from typing import List, Dict, Any, Optional

# ============================================================
# Simulated MCP Context (like ctx.info in real MCP)
# ============================================================

class SimulatedContext:
    """
    This class simulates the MCP 'context' object.
    In real MCP, tools can use ctx.info(...) to send progress notifications.
    Here, we simply print messages with a [NOTIFICATION] tag.
    """

    def info(self, message: str) -> None:
        """
        Display a progress message to the user and simulate a short delay.
        """
        # Print the notification in a formatted way
        print(f"   [NOTIFICATION] {message}")
        # Sleep for half a second to simulate some ongoing process
        time.sleep(0.5)


# ============================================================
# Tool 1: Simulated Web Search
# ============================================================

def simulated_web_search(topic: str) -> Dict[str, Any]:
    """
    Simulate a web search tool.
    Instead of connecting to the internet, it always returns a fixed set of articles.
    """
    print(f"   -> Tool 'simulated_web_search' called for: '{topic}'")

    # Define 3 mock articles (each is a dictionary with title, URL, and text).
    # This represents what a real "search API" might return.
    found_articles: List[Dict[str, str]] = [
        {
            "title": "Secrets of Proper Hydration",
            "url": "http://fakesite.com/hydration",
            "text": (
                "Drinking water is essential for the human body. "
                "It helps regulate body temperature, transport nutrients, and eliminate toxins. "
                "Remember to sip water consistently throughout the day."
            ),
        },
        {
            "title": "The Importance of Sleep for Energy",
            "url": "http://fakesite.com/sleep",
            "text": (
                "Quality sleep is crucial for physical and mental recovery. "
                "It directly affects mood, concentration, and the immune system. "
                "Building a consistent sleep routine can greatly improve daily life."
            ),
        },
        {
            "title": "Micro-Movements for Health",
            "url": "http://fakesite.com/micro-moves",
            "text": (
                "Short movement breaks reduce risks linked to sedentary behavior. "
                "Even 2–3 minutes of walking or stretching boost circulation and focus. "
                "Set gentle reminders to stand up every hour."
            ),
        },
    ]
    # Return dictionary with key "results" for consistency
    return {"results": found_articles}


# ============================================================
# Helper function: Extract first sentence from text
# ============================================================

def _first_sentence(text: str) -> Optional[str]:
    """
    Extract the first sentence of a paragraph.
    Ensures the sentence ends with a period.
    """
    if not text or not text.strip():
        return None

    # Split the text by ". " (basic heuristic for sentence boundaries)
    parts = text.strip().split(". ")
    first = parts[0].strip()

    if not first:
        return None

    # Add period if it’s missing
    if not first.endswith("."):
        first += "."
    return first


# ============================================================
# Tool 2: Simulated LLM Summary Generator
# ============================================================

def simulated_llm_summary(
    briefing_topic: str,
    article_list: List[Dict[str, str]],
    ctx: SimulatedContext,
) -> Dict[str, str]:
    """
    Simulate a Large Language Model (LLM) that generates a briefing from articles.
    This tool also sends progress notifications through ctx.info.
    """
    # Notify start of the summary process
    ctx.info(f"Starting summary for topic: '{briefing_topic}'")

    # Simulate the LLM "thinking"
    time.sleep(1)

    # Initialize briefing text with title and section header
    summary_text = f"Briefing (Simulated LLM) on: {briefing_topic}\n\n"
    summary_text += "Key Points (Generated):\n"

    # Store the list of sources for later reference
    sources_for_llm: List[str] = []

    # Process each article in the provided list
    for i, article in enumerate(article_list):
        title = article.get("title", f"Article {i+1}")
        url = article.get("url", "N/A")
        text = article.get("text", "")

        # Notify progress: which article is currently being processed
        ctx.info(f"Analyzing article {i+1}/{len(article_list)}: {title}")

        # Extract the first sentence of the article
        first_sentence = _first_sentence(text)
        if first_sentence:
            summary_text += f"- {first_sentence} [Source {i+1}]\n"
        else:
            summary_text += f"- (No content available) [Source {i+1}]\n"

        # Add this article’s info to the sources list
        sources_for_llm.append(f"[Source {i+1}] {title} - {url}")

        # Short delay to simulate incremental analysis
        time.sleep(0.3)

    # Append all sources at the end of the summary
    summary_text += "\nSources:\n" + "\n".join(sources_for_llm)

    # Final notification: summary is ready
    ctx.info("Summary complete.")

    return {"full_briefing": summary_text}


# ============================================================
# Tool 3: Save briefing to a text file
# ============================================================

def save_file(file_name: str, briefing_content: str) -> Dict[str, Optional[str]]:
    """
    Save the briefing into a folder called 'generated_briefings'.
    Creates the folder if it does not exist.
    """
    output_folder = "generated_briefings"
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    full_path = os.path.join(output_folder, file_name)
    try:
        # Open the file in write mode with UTF-8 encoding
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(briefing_content)

        print(f"   -> Tool 'save_file': briefing saved to '{full_path}'")
        return {"path": full_path}
    except Exception as e:
        print(f"Error while saving briefing: {e}")
        return {"path": None}


# ============================================================
# Tool 4: Simulate MCP Tool Discovery
# ============================================================

def simulated_tool_discovery() -> Dict[str, Any]:
    """
    Simulate the MCP "tool discovery".
    Returns the list of available tools with their descriptions.
    """
    print("   -> Tool 'simulated_tool_discovery' called.")

    # The "server" exposes these tools to the "client".
    available_tools = [
        {"name": "simulated_web_search", "description": "Searches for simulated articles on a topic."},
        {"name": "simulated_llm_summary", "description": "Generates a simulated LLM summary with progress notifications."},
        {"name": "save_file", "description": "Saves the briefing into a text file."},
    ]
    return {"tools": available_tools}


# ============================================================
# Helper: Sanitize topic into safe filename
# ============================================================

def _sanitize_filename(topic: str) -> str:
    """
    Convert a user topic into a safe filename string.
    Removes punctuation, accents, and extra spaces.
    """
    base = (topic or "").lower()
    # Remove punctuation except spaces, underscores, hyphens
    base = re.sub(r"[^\w\s-]", "", base, flags=re.UNICODE)
    # Replace whitespace with underscores
    base = re.sub(r"\s+", "_", base).strip("_")
    return base or "briefing"


# ============================================================
# Main Orchestration (Simulated MCP Client)
# ============================================================

if __name__ == "__main__":
    print("--- Simulated MCP Client Startup ---")

    # Step 1: Handshake (simulate connection setup)
    print("\nMCP SIMULATION: Performing handshake (establishing connection)...")
    time.sleep(1)  # Short delay to simulate handshake
    print("MCP SIMULATION: Handshake complete. Session established.")

    # Step 2: Tool discovery
    print("\nMCP SIMULATION: Requesting available tools from 'server'...")
    discovery_results = simulated_tool_discovery()
    print("MCP SIMULATION: Tools discovered:")
    for tool in discovery_results["tools"]:
        print(f" - {tool['name']}: {tool['description']}")

    # Step 3: Ask the user for a topic
    user_topic = input("\nWhat topic do you want a briefing on (e.g., 'urban ecology')? ").strip()
    if not user_topic:
        user_topic = "everyday health"
        print(f"No topic entered, using default: '{user_topic}'")

    print(f"\nThe bot will now generate a briefing for: '{user_topic}'")

    # Step 4: Call search tool
    print("\n1) Calling 'simulated_web_search'...")
    search_results = simulated_web_search(user_topic)
    articles = search_results.get("results", [])

    if not articles:
        print("   -> No simulated articles found. Stopping process.")
    else:
        print(f"   -> {len(articles)} simulated articles found.")

        # Step 5: Call LLM summarizer
        print("\n2) Calling 'simulated_llm_summary' (watch the NOTIFICATIONS!) ...")
        context_for_llm = SimulatedContext()  # New context for notifications
        summary_results = simulated_llm_summary(user_topic, articles, context_for_llm)
        briefing_content = summary_results.get("full_briefing", "Error: content not generated.")

        # Step 6: Save briefing
        print("\n3) Calling 'save_file'...")
        today = datetime.date.today().isoformat()
        safe_filename = _sanitize_filename(user_topic)
        briefing_filename = f"briefing_{safe_filename}_{today}.txt"

        save_results = save_file(briefing_filename, briefing_content)
        final_path = save_results.get("path")

        # Step 7: Final status
        print("\n--- Simulated Intelligent Bot Process Complete ---")
        if final_path:
            print(f"Your briefing is ready! Open the file: '{final_path}'")
        else:
            print("An error occurred while creating the briefing.")
