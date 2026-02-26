from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

template_translate = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to English: \n {initial_text}"
)

template_summary = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text: \n {text}"
)

model_en = ChatOpenAI(model="gpt-5-mini", temperature=0)

translate = template_translate | model_en | StrOutputParser()
pipeline = { "text": translate} | template_summary | model_en | StrOutputParser()

result = pipeline.invoke({"initial_text": "A banda inglesa Iron Maiden é uma das maiores do planeta"})

print(result)