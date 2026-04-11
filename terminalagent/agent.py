import os
import json
import requests
from openai import OpenAI
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY    = os.getenv("NVIDIA_API_KEY")
SERPAPI_KEY       = os.getenv("SERPAPI_KEY")
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")

# NVIDIA NIM client (OpenAI-compatible)
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

MODEL        = "qwen/qwen3-next-80b-a3b-instruct"
MEMORY_FILE  = "memory_agent.json"
MAX_MESSAGES = 50


# Tool 1: Google Search via SerpAPI
def google_search(query: str) -> str:
    print(f"\n  [Searching Google for: {query}]")
    params = {
        "q": query,
        "num": 5,
        "api_key": SERPAPI_KEY,
        "engine": "google",
    }
    try:
        results = GoogleSearch(params).get_dict().get("organic_results", [])
    except Exception as e:
        print(f"  Search failed: {e}")
        return f"Search failed due to a network error: {e}"

    if not results:
        return "No results found."

    output = ""
    for i, r in enumerate(results, 1):
        output += f"{i}. {r.get('title', 'N/A')}\n"
        output += f"   Link: {r.get('link', 'N/A')}\n"
        output += f"   Summary: {r.get('snippet', 'N/A')}\n\n"
    return output


# Tool 2: Send to Telegram
def send_telegram(message: str) -> str:
    print(f"\n  [Sending to Telegram...]")
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return "Telegram is not configured. Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"

    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return "Message sent to Telegram successfully!"
        else:
            error_data = response.json() if response.text else {}
            error_msg  = error_data.get("description", response.text)
            return f"Failed to send message. Telegram error: {error_msg}"
    except requests.exceptions.RequestException as e:
        return f"Network error while sending to Telegram: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


# Tool definitions (OpenAI format)
tools = [
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": (
                "Search Google for real-time information. "
                "Use this when the user asks about current events, "
                "leads, news, people, companies, or anything needing "
                "up-to-date web information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on Google"
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
                "ONLY use this tool if the user explicitly asks to send "
                "something to Telegram. Do not use it otherwise."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message content to send to Telegram"
                    }
                },
                "required": ["message"]
            }
        }
    }
]


# Tool router
def run_tool(tool_name: str, arguments: dict) -> str:
    if tool_name == "google_search":
        return google_search(arguments["query"])
    elif tool_name == "send_telegram":
        return send_telegram(arguments["message"])
    else:
        return f"Unknown tool: {tool_name}"

# Memory: load / save / trim
def load_history() -> list:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            history = json.load(f)
        # Drop corrupt tool messages (missing tool_call_id)
        sanitized = [
            msg for msg in history
            if not (msg.get("role") == "tool" and not msg.get("tool_call_id"))
        ]
        print("  Memory loaded from previous session.")
        return sanitized
    else:
        print("  No memory found. Starting fresh.")
        return [
            {
                "role": "system",
                "content": (
                    "You are a powerful AI assistant and lead generation agent. "
                    "You have two tools: google_search and send_telegram. "
                    "Use google_search when you need real-time or current information. "
                    "Use send_telegram ONLY when the user explicitly asks to send something to Telegram. "
                    "Do NOT use send_telegram unless the user requests it. "
                    "For general knowledge questions, answer from memory without searching. "
                    "Be concise, accurate, and professional."
                )
            }
        ]


def save_history(chat_history: list):
    with open(MEMORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=2)


def trim_history(chat_history: list) -> list:
    if len(chat_history) > MAX_MESSAGES + 1:
        trimmed = [chat_history[0]] + chat_history[-MAX_MESSAGES:]
        print(f"\n  [Memory trimmed to last {MAX_MESSAGES} messages]")
        return trimmed
    return chat_history


# Core agentic loop
def process_message(user_input: str, chat_history: list) -> str:
    chat_history.append({"role": "user", "content": user_input})

    max_steps = 8
    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,
            messages=chat_history,
            tools=tools,
            tool_choice="auto"
        )

        choice        = response.choices[0]
        finish_reason = choice.finish_reason
        msg           = choice.message

        if finish_reason == "tool_calls" and msg.tool_calls:
            # Convert assistant message to a plain dict for serialization
            assistant_entry = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id":   tc.id,
                        "type": tc.type,
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]
            }
            chat_history.append(assistant_entry)

            # Run every tool call and append its result
            for tc in msg.tool_calls:
                arguments   = json.loads(tc.function.arguments)
                tool_result = run_tool(tc.function.name, arguments)
                chat_history.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      tool_result
                })

        else:
            # Final text reply
            final_reply = msg.content or ""
            chat_history.append({"role": "assistant", "content": final_reply})
            return final_reply

    return "Max steps reached — stopping to avoid infinite loop."


# Main interactive loop
def main():
    print("\n" + "=" * 55)
    print(" AI Terminal Agent  |  Powered by NVIDIA NIM")
    print(" Type 'quit' to exit  |  'clear' to reset memory")
    print("=" * 55 + "\n")

    chat_history = load_history()

    while True:
        try:
            user_input = input("\n  You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                chat_history = trim_history(chat_history)
                save_history(chat_history)
                print("\n  Memory saved. Goodbye!")
                break

            if user_input.lower() == "clear":
                if os.path.exists(MEMORY_FILE):
                    os.remove(MEMORY_FILE)
                chat_history = load_history()
                print("\n  Memory cleared. Starting fresh.")
                continue

            print("\n  Agent: ", end="", flush=True)
            reply = process_message(user_input, chat_history)
            print(reply)

            chat_history = trim_history(chat_history)
            save_history(chat_history)

        except KeyboardInterrupt:
            chat_history = trim_history(chat_history)
            save_history(chat_history)
            print("\n\n  Memory saved. Goodbye!")
            break


if __name__ == "__main__":
    main()
