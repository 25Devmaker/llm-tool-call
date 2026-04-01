import requests
import json
from groq import Groq
from serpapi import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv()

# api keys
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
SERPAPI_KEY      = os.getenv("SERPAPI_KEY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")

# groq client
client = Groq(api_key=GROQ_API_KEY)


#Tool 1: Google Search 
def google_search(query: str) -> str:
    print(f"\n[Tool 1] Google Search called with query: {query}")
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


#Tool 2: Send Telegram 
def send_telegram(message: str) -> str:
    print(f"\n[Tool 2] Sending to Telegram...")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    response = requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    })

    if response.status_code == 200:
        return "Message sent successfully."
    else:
        return f"Failed to send message. Error: {response.text}"


#tool descriptions (outside any function!) 
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
                "Send a message to user's Telegram. "
                "Use this AFTER google_search to send the formatted results."
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


#Tool router 
def run_tool(tool_name: str, arguments: dict) -> str:
    if tool_name == "google_search":
        return google_search(arguments["query"])
    elif tool_name == "send_telegram":
        return send_telegram(arguments["message"])
    else:
        return f"Unknown tool: {tool_name}"


#main function 
def ask(user_message: str):
    print(f"\n You: {user_message}")
    print("=" * 55)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a lead generation assistant. "
                "You have ONLY TWO tools: google_search and send_telegram. "
                "STRICT RULES: "
                "1. Call google_search ONLY ONCE. "
                "2. After getting search results, IMMEDIATELY call send_telegram. "
                "3. Do NOT search multiple times. "
                "4. Format the results nicely before sending to Telegram. "
                "5. Do not attempt to open URLs or browse websites."
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    step = 1
    max_steps = 10
    tools_used = []      # track which tools have already been called
    last_result = ""     # store last tool result

    while step <= max_steps:
        print(f"\n --- Step {step}: Asking Groq what to do next ---")

        # if google_search already ran, force send_telegram directly
        # don't even ask Groq — just send it
        if "google_search" in tools_used and "send_telegram" not in tools_used:
            print(" Google search done — forcing Telegram send now...")
            result = send_telegram(f"Here are your leads:\n\n{last_result}")
            print(f" {result}")
            break

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        finish_reason = response.choices[0].finish_reason

        if finish_reason == "tool_calls":
            tool_call  = response.choices[0].message.tool_calls[0]
            tool_name  = tool_call.function.name
            arguments  = json.loads(tool_call.function.arguments)

            print(f" Groq wants to use: {tool_name}")
            print(f" With arguments: {arguments}")

            tool_result = run_tool(tool_name, arguments)
            last_result = tool_result          # save result for forced Telegram send
            print(f" Tool result: {tool_result[:100]}...")

            tools_used.append(tool_name)       # track this tool was used

            messages.append(response.choices[0].message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

            step += 1

        else:
            final_answer = response.choices[0].message.content
            print(f"\n Groq's final answer:\n{final_answer}")
            break


#run it 
if __name__ == "__main__":
    ask("Find me 5 leads who are actively hiring for model photoshoots or automation setup, and send the results to my Telegram")