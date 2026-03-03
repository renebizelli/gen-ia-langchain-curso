from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

load_dotenv()

@tool("calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Evaluate a  simple mathematic expression and returns the result."""
    try:
        result = eval(expression)
    except Exception as e:
        return f"Error {e}"

    return str(result)


@tool("web_search_mock", return_direct=True)
def web_search_mock(query: str) -> str:
    """Mock web search tool. Returns a hardcoded result."""   

    data = {"Brazil": "Brasilia", "France" : "Paris", "Germany" : "Berlim", "Italy":"Rome", "Spain" : "Madrid"}

    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"The capital of {country} is {capital}."

    return "I don't know the capital of that country."

tools = [calculator, web_search_mock]

llm = ChatOpenAI(model="gpt-5-mini", disable_streaming=True).bind_tools(tools,
    tool_choice={
        "type": "function",
        "function": {"name": "web_search_mock"}
    }
)


prompt = PromptTemplate.from_template(
    """
    You must use tools to answer.
    If the tool says it does not know, you must respond:
    'I don't know.'
    Do not call the tool again.

    {tools}

    Use the following format:

    Question: the input question you must answer.
    Thought: you should always think about  what to do
    Action: the action to take, should be one of [{tool_names}]
    Action input: the input to the action
    Observation: the result  of the action

    ...(this Thought/Action/Action input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final  answer to the original input question

    Rules:
    - If you choose an Action, do NOT include Final answer in the same step
    - After Action and Action Input, stop and wait of Observation
    - Never search the internet. Only use the tools provided.

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """
)

agent_chain = create_agent(llm, tools)

result = agent_chain.invoke(
    {"messages": [
    {"role": "user", "content": "What is the capital of Irseddean?"}
    ]}
    , config={ "recursion_limit": 2 })
#print(agent_executor.invoke({"input": "How much is 10 + 10"}))    

for message in result["messages"]:
    if isinstance(message, AIMessage) and message.content.strip():
        print(message.content)
