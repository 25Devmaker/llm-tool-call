# HOW THIS WORKS
# 1. You send a message to Groq (the LLM)
# 2. Groq reads your message AND the tool description
# 3. Groq decides: "do I need to search? yes/no"
# 4. If yes → Groq tells Python WHAT to search
# 5. Python runs the actual google_search() function
# 6. Results go back to Groq
# 7. Groq reads the results and gives you a final answer


from groq import Groq
from serpapi import GoogleSearch
from dotenv import load_dotenv
import os
# API keys 
GROQ_API_KEY   = os.getenv("your_groq_api_key")    
SERPAPI_KEY    = os.getenv("your_serpapi_key")      

#Step 1: Create the Groq client 
# This is like opening a connection to the Groq LLM
client = Groq(api_key=GROQ_API_KEY)


#  Step 2: Write the actual tool (plain Python function) 
# This code, Groq never runs this directly.
# Groq just tells YOU when to run it and with what arguments.
def google_search(query: str) -> str:
    """Search Google using SerpAPI and return results as text."""
    params = {
        "q": query,
        "num": 5,
        "api_key": SERPAPI_KEY,
        "engine": "google",
    }
    search  = GoogleSearch(params)
    results = search.get_dict().get("organic_results", [])

    # Format results into a readable string so Groq can understand them
    if not results:
        return "No results found."

    output = ""
    for i, r in enumerate(results, 1):
        output += f"{i}. {r.get('title', 'N/A')}\n"
        output += f"   Link: {r.get('link', 'N/A')}\n"
        output += f"   Summary: {r.get('snippet', 'N/A')}\n\n"
    return output


# Step 3: Describe the tool to Groq
# You are telling Groq: "hey, you have access to this tool,
# here is what it does and what arguments it needs"
# Groq reads this description to decide WHEN to use it.
tools = [
    {
        "type": "function",
        "function": {
            "name": "google_search",           # must match your function name above
            "description": (
                "Search Google to find real-time information. "
                "Use this when the user asks to find leads, clients, "
                "companies, news, or anything that needs a web search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on Google"
                    }
                },
                "required": ["query"]          # query is mandatory
            }
        }
    }
]


#  Step 4: The main function that ties everything together 
def ask(user_message: str):
    print(f"\n You: {user_message}")
    print("-" * 50)

    # Send user message + tool description to Groq
    # Groq will either reply directly OR ask to call a tool
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that finds freelance clients. "
                "When the user asks to find clients or search for something, "
                "always use the google_search tool."
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",   # fast free model on Groq
        messages=messages,
        tools=tools,
        tool_choice="auto"        # "auto" = Groq decides whether to use a tool or not
    )

    # Step 5: Check if Groq wants to call a tool
    # response.choices[0].finish_reason tells us WHY Groq stopped
    # "tool_calls" = Groq wants to use a tool
    # "stop"       = Groq just replied with text, no tool needed
    finish_reason = response.choices[0].finish_reason

    if finish_reason == "tool_calls":
        # Groq decided to use the tool
        tool_call = response.choices[0].message.tool_calls[0]

        # What function does Groq want to call?
        function_name = tool_call.function.name
        print(f" Groq decided to use tool: {function_name}")

        # What arguments is Groq passing to the function?
        import json
        arguments = json.loads(tool_call.function.arguments)
        query = arguments["query"]
        print(f" Searching for: {query}\n")

        # Step 6: YOU run the actual function
        # Groq told you what to search — now you actually do it
        search_results = google_search(query)
        print(f" Raw search results:\n{search_results}")

        # Step 7: Send results back to Groq 
        # Now Groq reads the real results and gives a final answer
        messages.append(response.choices[0].message)   # Groq's tool call message
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": search_results                  # actual results from SerpAPI
        })

        final_response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
        )

        final_answer = final_response.choices[0].message.content
        print(f"\n Groq's final answer:\n{final_answer}")

    else:
        # Groq didn't need a tool, just replied with text directly
        direct_answer = response.choices[0].message.content
        print(f" Groq (no tool needed): {direct_answer}")


# Step 8: Run it
if __name__ == "__main__":
    # This should trigger the tool (needs a web search)
    ask("find me clients who are actively hiring for automation setup or AI model photoshoots")

    # This should NOT trigger the tool (Groq can answer from its own knowledge)
    ask("What is Python?")