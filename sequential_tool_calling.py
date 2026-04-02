# ============================================================
# SEQUENTIAL TOOL CALLING
#
# How this is different from single tool calling:
# - Before: Groq used ONE tool and stopped
# - Now:    Groq uses tool 1 (search) → reads results →
#           decides to use tool 2 (telegram) → sends leads
#
# Groq figures out the order on its own.
# You just say "find clients and send to telegram"
# and it chains the tools automatically.
#
# Flow:
# 1. You ask: "find leads and send to telegram"
# 2. Groq calls google_search("...")
# 3. Python runs google_search → gets results
# 4. Results sent back to Groq
# 5. Groq reads results → calls send_telegram("...")
# 6. Python runs send_telegram → message sent
# 7. Groq gives final confirmation
# ============================================================

import json
import requests
from groq import Groq
from serpapi import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv()

# ── Your API keys ──────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
SERPAPI_KEY    = os.getenv("SERPAPI_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ── Groq client ────────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)


# ══════════════════════════════════════════════════════════
# TOOL 1: Google Search
# Same as before — searches Google and returns results
# ══════════════════════════════════════════════════════════
def google_search(query: str) -> str:
    print(f"\n [Tool 1] Searching Google for: {query}")
    params = {
        "q": query,
        "num": 5,
        "api_key": SERPAPI_KEY,
        "engine": "google",
    }
    results = GoogleSearch(params).get_dict().get("organic_results", [])

    if not results:
        return "No results found."

    output = ""
    for i, r in enumerate(results, 1):
        output += f"{i}. {r.get('title', 'N/A')}\n"
        output += f"   Link: {r.get('link', 'N/A')}\n"
        output += f"   Summary: {r.get('snippet', 'N/A')}\n\n"
    return output


# ══════════════════════════════════════════════════════════
# TOOL 2: Send Telegram
# New tool — sends a message to your Telegram bot
# ══════════════════════════════════════════════════════════
def send_telegram(message: str) -> str:
    print(f"\n [Tool 2] Sending to Telegram...")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    response = requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"   # lets you use *bold* and _italic_ in messages
    })

    if response.status_code == 200:
        return "Message successfully sent to Telegram!"
    else:
        return f"Failed to send. Error: {response.text}"


# ══════════════════════════════════════════════════════════
# TOOL DESCRIPTIONS
# You are describing BOTH tools to Groq so it knows
# what tools are available and when to use each one
# ══════════════════════════════════════════════════════════
tools = [
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": (
                "Search Google to find real-time information. "
                "Use this to find leads, clients, companies, or any web info. "
                "Always use this FIRST before sending to Telegram."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_telegram",
            "description": (
                "Send a message to the user's Telegram. "
                "Use this AFTER google_search to deliver the results. "
                "Format the leads clearly before sending."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The formatted message to send to Telegram"
                    }
                },
                "required": ["message"]
            }
        }
    }
]


# ══════════════════════════════════════════════════════════
# TOOL ROUTER
# This function looks at what tool Groq wants to call
# and runs the matching Python function
# This is how you handle MULTIPLE tools cleanly
# ══════════════════════════════════════════════════════════
def run_tool(tool_name: str, arguments: dict) -> str:
    if tool_name == "google_search":
        return google_search(arguments["query"])
    elif tool_name == "send_telegram":
        return send_telegram(arguments["message"])
    else:
        return f"Unknown tool: {tool_name}"


# ══════════════════════════════════════════════════════════
# MAIN FUNCTION
# The key difference from single tool calling:
# We use a LOOP — keep checking if Groq wants another tool
# until it finally says "stop" (task complete)
# ══════════════════════════════════════════════════════════
def ask(user_message: str):
    print(f"\n You: {user_message}")
    print("=" * 55)

    # Start the conversation
    messages = [
        {
            "role": "system",
            "content": (
                "You are a lead generation assistant. "
                "You have access to ONLY TWO tools: google_search and send_telegram. "
                "Do NOT use any other tools. "
                "When asked to find clients: "
                "Step 1 - call google_search with a relevant query. "
                "Step 2 - call send_telegram with the formatted results. "
                "Do not attempt to open URLs or browse websites."
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    # ── The sequential loop ────────────────────────────────
    # This is the core difference from single tool calling.
    # We keep looping until Groq says finish_reason = "stop"
    # meaning it has finished using all the tools it needed.
    #
    # Round 1: Groq calls google_search
    # Round 2: Groq calls send_telegram
    # Round 3: Groq says "stop" → done
    
    step = 1
    max_steps = 5  # safety to prevent infinite loops
    while step <= max_steps:
        print(f"\n --- Step {step}: Asking Groq what to do next ---")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        finish_reason = response.choices[0].finish_reason

        if finish_reason == "tool_calls":
            # Groq wants to use a tool
            tool_call     = response.choices[0].message.tool_calls[0]
            tool_name     = tool_call.function.name
            arguments     = json.loads(tool_call.function.arguments)

            print(f" Groq wants to use: {tool_name}")
            print(f" With arguments: {arguments}")

            # Run the tool
            tool_result = run_tool(tool_name, arguments)
            print(f" Tool result: {tool_result[:100]}...")  # print first 100 chars

            # Add Groq's tool call message to history
            messages.append(response.choices[0].message)

            # Add the tool result back to history so Groq can read it
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

            step += 1

        else:
            # finish_reason == "stop" — Groq is done with all tools
            print("Max steps reached — stopping.")
            break   # exit the loop


# ── Run it ─────────────────────────────────────────────────
if __name__ == "__main__":
    ask("Find me 5 freelance web design clients who are actively hiring and send the results to my Telegram")