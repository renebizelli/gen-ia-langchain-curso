from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

question_template = PromptTemplate(
    input_variables=["name"],
    template="Hello, {name}!"
)

text = question_template.format(name="René")

print(text)

model = ChatOpenAI(model="gpt-5-nano", temperature=0.5)

chain = question_template | model

result = chain.invoke({"name": "John"})

print(result.content)










