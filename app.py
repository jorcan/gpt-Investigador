import os
from dotenv import load_dotenv
import streamlit as st
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage

# Cargar las variables de entorno desde el archivo .env
load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# 1. Herramienta para búsqueda
def search(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query
    })
    print(payload)
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    
    print(response)

    return response.text
    


# 2. Herramienta para scraping
def scrape_website(objective: str, url: str):
    # Extraer contenido de un sitio web y resumirlo si es necesario
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    data = {
        "url": url
    }

    data_json = json.dumps(data)
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

# 3. Resumen del contenido
def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Escribe un resumen del siguiente texto para {objective}:
    "{text}"
    RESUMEN:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

# Definir la estructura de entrada para la herramienta de scraping
class ScrapeWebsiteInput(BaseModel):
    """Entradas para la función de scraping"""
    objective: str = Field(
        description="El objetivo y tarea que los usuarios dan al agente")
    url: str = Field(description="La URL del sitio web a ser escrapeado")

# Definir la herramienta de scraping
class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "Útil para obtener datos de una URL de un sitio web y, posiblemente, resumir el contenido si es largo."
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error aquí")

# Crear una lista de herramientas para el agente
tools = [
    Tool(
        name="Busqueda",
        func=search,
        description="Útil para responder preguntas sobre eventos actuales y datos. Hacer preguntas específicas es recomendado."
    ),
    ScrapeWebsiteTool(),
]

# Definir el mensaje del sistema para el agente
system_message = SystemMessage(
    content="""Eres un investigador de clase mundial, capaz de realizar investigaciones detalladas sobre cualquier tema y producir resultados basados en hechos; no inventas cosas, te esfuerzas al máximo para recopilar hechos y datos que respalden la investigación.
            
            Asegúrate de cumplir el objetivo anterior con las siguientes reglas:

1/ Debes realizar suficiente investigación para recopilar la mayor cantidad posible de información sobre el objetivo.
2/ Si hay URLs de enlaces y artículos relevantes, los recopilarás para obtener más información.
3/ Después de recopilar y buscar información, debes preguntarte "¿hay cosas nuevas que debería buscar y recopilar en función de los datos que he recolectado para mejorar la calidad de la investigación?" Si la respuesta es sí, continúa; pero no hagas esto más de 3 iteraciones.
4/ No debes inventar cosas, solo debes escribir hechos y datos que hayas recopilado.
5/ En el resultado final, debes incluir todos los datos de referencia y enlaces para respaldar tu investigación.
7/ En el resultado final, debes incluir todos los datos de referencia y enlaces para respaldar tu investigación; debes incluir todos los datos de referencia y enlaces para respaldar tu investigación."""
)

# Configuraciones adicionales para el agente
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=2000)

# Inicializar el agente
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

# 4. Usar Streamlit para crear una aplicación web
def main():
    
    # Configuración de la página y título
    st.set_page_config(page_title="AI BrainQ Agente", page_icon=":male_detective:")
    # Cargar la imagen del logo
    image_path = os.path.join(os.path.dirname(__file__), "static/images/brainq-logo.png")
    st.image(image_path, width=200)    
    

    # Encabezado y entrada de texto
    st.header("AI Agente de investigación 🕵🏻")
    query = st.text_input("Objetivo de investigación")

    if query:
        st.write("Estoy investigando...   ", query)

        # Llamar al agente
        result = agent({"input": query})

        # Mostrar resultado
        if "output" in result:
            st.info(result['output'])
        else:
            st.error("No se encontró el resultado esperado en la respuesta del agente.")

if __name__ == '__main__':
    # Definir el puerto 8080 para Streamlit
    #port = 8080
    #st.set_option("server.port", port)

    # Ejecutar la función principal
    main()


#5. Set this as an API endpoint via FastAPI
''' app = FastAPI()


class Query(BaseModel):
    query: str


@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    return actual_content '''
