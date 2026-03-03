from math import dist
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

def prepare_inputs(payload: dist) -> dist:
    raw_history = payload.get("raw_history", [])
    trimmed = trim_messages(raw_history, 
                            token_counter=len, 
                            max_tokens=2,  #nao sabe meu nome por conta dessa config. ele so pega as 2 ultimas msgs.
                            strategy="last", 
                            start_on="human",
                            include_system=True,
                            allow_partial=False)  
    
    return {
        "input": payload.get("input", ""),
        "history": trimmed
    }

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant than answer with a short joke when possible."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


prepare = RunnableLambda(prepare_inputs)

chain = prepare | prompt | ChatOpenAI(model="gpt-5-nano", temperature=0.9)   

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="raw_history"
)

config = {"configurable" :  { "session_id": "user_123"}} 

response1 = conversational_chain.invoke({"input": "Hello, my name is René. Replay with 'OK' and do not mention my name"}, config=config)
print("Response 1:", response1.content)

response2 = conversational_chain.invoke({"input": "Tell me a one-sentence fun fact, Do not mention my name"}, config=config)
print("Response 2:", response2.content)

response3 = conversational_chain.invoke({"input": "What is myname?"}, config=config)
print("Response 3:", response3.content)