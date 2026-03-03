from langchain_core.runnables import chain, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

long_text = """
A disciplina é a ponte entre intenção e realização.
Todos querem resultados, mas poucos sustentam o processo necessário.
O problema não é falta de talento, é falta de constância.
Clareza de objetivo elimina distrações desnecessárias.
Se você acorda sem prioridade definida, o dia controla você.
Grandes projetos nascem de pequenas entregas bem feitas.
Errar faz parte, repetir o erro por descuido é escolha.
A comparação constante rouba energia que deveria ser usada para evoluir.
Foque em melhorar 1% por dia e observe o efeito acumulado.
O ambiente influencia mais do que a motivação momentânea.
Organize sua agenda antes que o caos organize por você.
Dizer “não” é proteger seu foco e seu tempo.
Energia é recurso limitado, use com estratégia.
Aprender continuamente é a única forma de não ficar obsoleto.
Mentores encurtam caminhos, mas você precisa caminhar.
Feedback honesto é desconfortável, porém valioso.
Evite a armadilha de começar muito e terminar pouco.
Consistência vence intensidade esporádica.
Pequenos hábitos diários constroem identidades sólidas.
Planejamento sem ação é apenas intenção elegante.
Ação sem planejamento é esforço mal direcionado.
Equilíbrio é ajuste constante, não estado permanente.
Quando algo não funciona, ajuste a estratégia, não o sonho.
Relacionamentos são ativos que exigem cuidado contínuo.
Comunicação clara evita conflitos desnecessários.
Coragem não é ausência de medo, é decisão apesar dele.
Celebre avanços, mas não se acomode neles.
Lembre-se de que resultados levam tempo para amadurecer.
Seu padrão atual determina seu próximo nível.
Comece hoje, mesmo que seja pequeno, mas comece.
""".replace("\n", "")

# Prompt
prompt = PromptTemplate.from_template("""
Resuma o texto abaixo de forma clara e objetiva:
{text}
""")

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

parts = splitter.create_documents([long_text])

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

map_prompt = PromptTemplate(
    input_variables=["text"],
    template="Faça um resumo conciso do texto a seguir: {text}")

map_chain = map_prompt | model | StrOutputParser()

prepare_map_inputs = RunnableLambda(lambda docs: [{"text" : d.page_content} for d in docs])
map_stage = prepare_map_inputs | map_chain.map()

reduce_prompt= PromptTemplate(
    input_variables=["text"],
    template="Combine o seguinte resumo em apenas um resumo geral: {text}"
)
reduce_chain = reduce_prompt | model | StrOutputParser()

prepare_reduce_input = RunnableLambda(lambda summaries: {"text": "\n".join(summaries)})
pipeline = map_stage | prepare_reduce_input | reduce_chain

result = pipeline.invoke(parts)