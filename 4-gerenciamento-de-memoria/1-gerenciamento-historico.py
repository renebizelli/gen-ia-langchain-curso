from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

load_dotenv()


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chat_model = ChatOpenAI(model="gpt-5-nano", temperature=0.9) 

chain = prompt | chat_model

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

config = {"configurable" :  { "session_id": "user_123"}} 

response1 = conversational_chain.invoke({"input": "Hello, my name is René. How are you?"}, config=config)
print("Response 1:", response1.content)

response2 = conversational_chain.invoke({"input": "Can you repeat my name?"}, config=config)
print("Response 2:", response2.content)

response3 = conversational_chain.invoke({"input": "Can you repeat my name in a motivational way?"}, config=config)
print("Response 3:", response3.content)