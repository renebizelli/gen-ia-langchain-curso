from typing import Any


from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain, RunnableLambda

load_dotenv()

def square(x:dict) -> dict:
    return { "result": x["value"] * x["value"] }

question_template = PromptTemplate(
    input_variables=["result"],
    template="Quanto é a {result} dividido por 2?"
)

model = ChatOpenAI(model="gpt-5-nano", temperature=0.5)

square_lambda = RunnableLambda(square)

chain = square_lambda | question_template | model

result = chain.invoke({"value" : 10})

print(result.content)







