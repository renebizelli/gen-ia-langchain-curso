from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

long_text = """
A harmonia musical é o ramo da música que estuda a combinação dos sons tocados simultaneamente. Ela está diretamente ligada aos acordes e à forma como eles se encadeiam ao longo de uma composição. Por meio da harmonia, a música ganha cor emocional, criando sensações de tensão, repouso, alegria ou tristeza. Enquanto a melodia se desenvolve horizontalmente, a harmonia atua de forma vertical, sustentando e enriquecendo o som. As regras harmônicas ajudam o compositor a escolher acordes que soem coerentes entre si. Ao mesmo tempo, a quebra dessas regras pode gerar efeitos expressivos e criativos. A harmonia está presente em praticamente todos os estilos musicais, do clássico ao popular.
Compreendê-la é essencial para quem deseja compor, arranjar ou interpretar música com mais consciência e profundidade.
"""

# Prompt
prompt = PromptTemplate.from_template("""
Resuma o texto abaixo de forma clara e objetiva:
{text}
""")

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

parts = splitter.create_documents([long_text])

model = ChatOpenAI( model="gpt-5-nano", temperature=0)

map_chain = prompt | model | StrOutputParser()

summaries = [
    map_chain.invoke({"text" : part.page_content})
    for part in parts
]
 

final_summary = map_chain.invoke({"text" : "\n".join(summaries)})

print(final_summary)