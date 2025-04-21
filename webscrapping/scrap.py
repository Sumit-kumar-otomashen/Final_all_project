import os
import uuid
from typing import List, Annotated
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import TavilySearchResults

# Set environment variables
os.environ["TAVILY_API_KEY"] = "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"
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"  # Optional: to avoid the USER_AGENT warning

# Define the Web Scraping Tool with RunnableConfig
@tool
def scrape_webpages(urls: List[str], config: RunnableConfig) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    try:
        loader = WebBaseLoader(urls)
        docs = loader.load()
        return "\n\n".join(
            [
                f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
                for doc in docs
            ]
        )
    except Exception as e:
        return f"Error scraping web pages: {str(e)}"

# Instantiate the Tavily Search Tool
tavily_tool = TavilySearchResults(
    max_results=5,
    include_answer=True,
    include_raw_content=True,
    include_images=True
)

# List of tools
tools = [scrape_webpages, tavily_tool]

# Set up the Language Model
llm = ChatGroq(
    model="qwen-qwq-32b",
    api_key="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"  # Replace with your Grok API key
)

# System Prompt for Both Tools
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI assistant designed to help users by either scraping specific web pages or searching the web for information. You have access to two tools:

- **scrape_webpages**: Use this tool when the user provides specific URLs to scrape. It extracts detailed information from the provided web pages.
- **tavily_search**: Use this tool when the user asks a question or provides a query that requires searching the web. It returns search results in JSON format.

### Instructions:
1. Analyze the user's message to determine their intent.
2. If the user provides specific URLs and asks to scrape them, use the `scrape_webpages` tool with the list of URLs.
3. If the user asks a question or provides a query without specifying URLs, use the `tavily_search` tool with the query.
4. Present the results clearly, formatting the output as needed.

### Additional Guidelines:
- If the user's request is unclear, ask for clarification.
- Handle tool errors by reporting them clearly to the user.
- For `tavily_search` results, parse the JSON and present relevant information (e.g., the answer or top results).
- Use markdown to format outputs for readability.
"""
        ),
        ("placeholder", "{messages}"),
    ]
)

# Bind the tools to the language model
assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools)

# Define the State
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str

# Assistant Class with Error Handling
class Assistant:
    def __init__(self, runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        max_retries = 20
        retry_count = 0
        
        while retry_count < max_retries:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("user_id", None)
            state = {**state, "user_info": passenger_id}
            
            try:
                result = self.runnable.invoke(state)
                if not result.tool_calls and (
                    not result.content or 
                    (isinstance(result.content, list) and not result.content[0].get("text"))
                ):
                    messages = state["messages"] + [HumanMessage(content="Please provide a complete response with tool calls or content.")]
                    state = {**state, "messages": messages}
                    retry_count += 1
                else:
                    if result.tool_calls:
                        last_message = state["messages"][-1] if state["messages"] else None
                        if last_message and isinstance(last_message, AIMessage) and "Error" in last_message.content:
                            messages = state["messages"] + [HumanMessage(content=f"Tool call failed: {last_message.content}. Adjust the input and retry.")]
                            state = {**state, "messages": messages}
                            retry_count += 1
                            continue
                    break
            except Exception as e:
                if "tool_use_failed" in str(e):
                    messages = state["messages"] + [HumanMessage(content=f"Tool use failed: {str(e)}. Verify arguments and retry.")]
                    state = {**state, "messages": messages}
                    retry_count += 1
                else:
                    print(f"Error in Assistant: {str(e)}")
                    messages = state["messages"] + [AIMessage(content=f"Technical issue: {str(e)}. Please try again later.")]
                    return {"messages": messages}
        
        if retry_count >= max_retries:
            messages = state["messages"] + [AIMessage(content="Failed to generate a response after multiple attempts.")]
            return {"messages": messages}
        
        return {"messages": state["messages"] + [result]}

# Set up the State Graph
builder = StateGraph(State)
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Run the Task
if __name__ == "__main__":
    config = {"configurable": {"user_id": "12345", "thread_id": str(uuid.uuid4())}}
    initial_state = {
        "messages": [
            HumanMessage(
                content=" scrap this https://www.imdb.com/chart/top/ ,, scrap this website and make table of top 2 movies with their ratings and year of release"
            )
        ]
    }
    result = graph.invoke(initial_state, config)
    print("\n### Execution Log ###")
    for message in result["messages"]:
        print(f"{message.type.upper()}: {message.content}")