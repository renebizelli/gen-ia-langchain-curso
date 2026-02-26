from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain

load_dotenv()

@chain
def square(x:dict) -> dict:
    return { "result": x["value"] * x["value"] }

question_template = PromptTemplate(
    input_variables=["result"],
    template="Quanto é a {result} dividido por 2?"
)

model = ChatOpenAI(model="gpt-5-nano", temperature=0.5)

chain = square | question_template | model

result = chain.invoke({"value" : 10})

print(result.content)










