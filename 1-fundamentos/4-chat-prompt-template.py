from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

system = ("system", "You are a {job} that can answer questions and help with tasks.")
user = ("human", "{question}")

chat_prompt_template = ChatPromptTemplate.from_messages([system, user])

variables = [
    {
        "job": "teacher",
        "question": "What is the capital of France?"
    },
    {
        "job": "developer",
        "question": "What is the capital of Germany?"
    }
]

chat_model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")


for variable in variables:
    messages = chat_prompt_template.format_messages(job=variable["job"], question=variable["question"])

    answer = chat_model.invoke(messages)
    print(answer.type, ":", answer.content)
    print(answer.id)
    print(answer)
