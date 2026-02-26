from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

template = PromptTemplate(
    input_variables=["name"],
    template="Hello, {name}!"
)

text = template.format(name="John")

print(text)










