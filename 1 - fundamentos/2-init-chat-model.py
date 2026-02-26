from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

chat_model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

answer = chat_model.invoke("Hello, world!")

print(answer.content)