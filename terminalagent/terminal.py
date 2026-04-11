import os
import json
import requests
from groq import Groq
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
SERPAPI_KEY      = os.getenv("SERPAPI_KEY")
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

client = Groq(api_key=GROQ_API_KEY)

MEMORY_FILE  = "memory.json"
MAX_MESSAGES = 50


# Tool 1: Google Search
def google_search(query: str) -> str:
    print(f"\n [Searching Google for: {query}]")
    params = {
        "q": query,
        "num": 5,
        "api_key": SERPAPI_KEY,
        "engine": "google",
    }
    try:
        results = GoogleSearch(params).get_dict().get("organic_results", [])
    except Exception as e:
        print(f" Search failed: {e}")
        return f"Search failed due to a network error: {e}"

    if not results:
        return "No results found."

    output = ""
    for i, r in enumerate(results, 1):
        output += f"{i}. {r.get('title', 'N/A')}\n"
        output += f"   Link: {r.get('link', 'N/A')}\n"
        output += f"   Summary: {r.get('snippet', 'N/A')}\n\n"
    return output


# Tool 2: Send Telegram
def send_telegram(message: str) -> str:
    print(f"\n [Sending to Telegram...]")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return "Telegram is not configured. Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return "Message sent to Telegram successfully!"
        else:
            error_data = response.json() if response.text else {}
            error_msg  = error_data.get("description", response.text)
            return f"Failed to send message. Error: {error_msg}"
    except requests.exceptions.RequestException as e:
        print(f" Network error while sending to Telegram: {e}")
        return f"Network error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


# Tool descriptions
tools = [
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": (
                "Search Google for real-time information. "
                "Use this when the user asks about current events, "
                "wants to find leads, clients, news, or anything "
                "that needs up-to-date web information."
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
                "Use this when the user asks to send something "
                "to Telegram or wants results delivered there."
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


# Memory functions
def load_history() -> list:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            print(" Memory loaded from previous session.")
            history = json.load(f)
        # Drop any tool messages that are missing tool_call_id (corrupt/old entries)
        sanitized = []
        for msg in history:
            if msg.get("role") == "tool" and not msg.get("tool_call_id"):
                continue
            sanitized.append(msg)
        return sanitized
    else:
        print(" No memory history found. Starting fresh.")
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant, lead generation agent, "
                    "and a senior professional coder in all programming languages. "
                    "You have two tools: google_search and send_telegram. "
                    "Use google_search when you need real-time or current information. "
                    "Use send_telegram when the user wants to send something to Telegram. "
                    "For general questions answer from your own knowledge — don't search unnecessarily. "
                    "Be concise and helpful. "
                    "Remember details the user shares about themselves."
                )
            }
        ]


def save_history(chat_history: list):
    serializable = []
    for m in chat_history:
        if hasattr(m, "role"):
            # It's a Groq message object (e.g. assistant with tool_calls)
            entry = {"role": m.role, "content": m.content}
            if hasattr(m, "tool_calls") and m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in m.tool_calls
                ]
        else:
            # It's already a plain dict (user, system, or tool messages)
            entry = dict(m)

        serializable.append(entry)

    with open(MEMORY_FILE, "w") as f:
        json.dump(serializable, f, indent=2)


def trim_history(chat_history: list) -> list:
    if len(chat_history) > MAX_MESSAGES + 1:
        trimmed = [chat_history[0]] + chat_history[-MAX_MESSAGES:]
        print(f"\n [Memory trimmed to last {MAX_MESSAGES} messages]")
        return trimmed
    return chat_history


# Process one message
def process_message(user_input: str, chat_history: list) -> str:
    chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    max_steps = 8
    step = 0

    while step < max_steps:
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=chat_history,
            tools=tools,
            tool_choice="auto"
        )

        finish_reason = response.choices[0].finish_reason

        if finish_reason == "tool_calls":
            msg = response.choices[0].message
            tool_calls = msg.tool_calls

            # Convert assistant message object to a plain dict for serialization
            assistant_entry = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            }
            chat_history.append(assistant_entry)

            # Process every tool call in the response
            for tool_call in tool_calls:
                tool_name   = tool_call.function.name
                arguments   = json.loads(tool_call.function.arguments)
                tool_result = run_tool(tool_name, arguments)
                chat_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

            step += 1

        else:
            final_reply = response.choices[0].message.content
            chat_history.append({
                "role": "assistant",
                "content": final_reply
            })
            return final_reply

    return "Max steps reached — stopping."


# Main interactive loop
def main():
    print("\n" + "=" * 55)
    print(" AI Terminal Agent — with memory history")
    print(" Type 'quit' to exit | 'clear' to reset memory")
    print("=" * 55 + "\n")

    chat_history = load_history()

    while True:
        try:
            user_input = input("\n You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                chat_history = trim_history(chat_history)
                save_history(chat_history)
                print("\n Memory saved. Goodbye!")
                break

            if user_input.lower() == "clear":
                if os.path.exists(MEMORY_FILE):
                    os.remove(MEMORY_FILE)
                chat_history = load_history()
                print("\n Memory cleared. Starting fresh.")
                continue

            print("\n Agent: ", end="", flush=True)
            reply = process_message(user_input, chat_history)
            print(reply)

            chat_history = trim_history(chat_history)
            save_history(chat_history)

        except KeyboardInterrupt:
            chat_history = trim_history(chat_history)
            save_history(chat_history)
            print("\n\n Memory saved. Goodbye!")
            break


if __name__ == "__main__":
    main()