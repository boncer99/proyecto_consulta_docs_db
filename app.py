#!pip install pymupdf langchain langchain-community langchain-elasticsearch langchain-openai
#!pip install python-dotenv
#!pip install langgraph
#!pip install langchain-elasticsearch

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore

#from langchain_core.prompts import ChatPromptTemplate ORIGINAL HASTA ANTES DE ESTE EJEMPLO
from langchain.prompts import ChatPromptTemplate   # este en su reemplazo porque funcionaba con esta libreria

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
#from langchain_community.document_loaders import PyMuPDFLoader
#from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

import sys
#from flask import Flask, jsonify, request
from langchain.tools import tool

from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from flask import Flask, request, render_template, jsonify

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit


load_dotenv()

# ========= CONFIGURACIÓN ==========

# Obtener las variables del entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")    # para llm = ChatOpenAI(openai_api_key=openai_api_key)
uribd = os.getenv("DB_ACCESS")

ES_URL = os.getenv("ES_URL")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
INDEX_NAME = os.getenv("INDEX_NAME")

# Establecer la clave de OpenAI como variable de entorno
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY    # no estaba originalmente. SACAR PARA RAILWAY

app = Flask(__name__)

# =============== conexion con la base de datos ==========================#
db_data = SQLDatabase.from_uri(uribd)
# Herramienta BD
toolkit_bd = SQLDatabaseToolkit(db=db_data,llm=ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key,verbose=True))
tools_bd = toolkit_bd.get_tools()

# ================= herramientas de seleccion de tablas =================================#
def seleccionar_tabla_v2(question: str) -> str:
    """
    Selecciona la tabla correcta basada en la pregunta utilizando la LLM para inferir el nombre de la tabla,
    y luego verifica si el nombre de la tabla existe en las tablas disponibles.
    """
    # Obtiene las tablas disponibles desde la base de datos
    tablas_disponibles = db_data.get_table_names()  # Listado dinámico de tablas

    # Crea un mensaje que pasa la lista de tablas disponibles al modelo de lenguaje
    tablas_str = ", ".join(tablas_disponibles)  # Las tablas disponibles como un string

    # Consultamos a la LLM para obtener el nombre de la tabla basado en la pregunta y las tablas disponibles
    prompt = (f"Las siguientes tablas están disponibles en la base de datos: {tablas_str}. "
              f"Segun las tablas indicadas  ¿Devuelve sólo el nombre de la tabla a  la que corresponde esa pregunta: '{question}'?")
    #model = ChatOpenAI(openai_api_key=openai_api_key,verbose=True)
    model = ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key,verbose=True)

    # Suponiendo que 'model' es el objeto que invoca el modelo de lenguaje
    #nombre_tabla_sugerido = model(prompt).strip().lower()  # Respuesta procesada a minúsculas
    respuesta = model.invoke([HumanMessage(content=prompt)])
    nombre_tabla_sugerido = respuesta.content.strip().lower()

    # Validamos si la tabla sugerida por el modelo existe en la base de datos
    #if nombre_tabla_sugerido not in tablas_disponibles:
    #    return f"[ERROR] La tabla '{nombre_tabla_sugerido}' no se reconoce. Las tablas disponibles son: {', '.join(tablas_disponibles)}"
    return nombre_tabla_sugerido
# Función para seleccionar la tabla correcta basada en la pregunta
def seleccionar_tabla(question: str) -> str:
    # Define las tablas disponibles
    tablas_disponibles = ['temperatura_registros', 'humedad_registros', 'calidad_aire_registros', 'sensores', 'historial_dispositivos']
    # Aquí puedes agregar lógica de selección, por ejemplo, buscar palabras clave en la pregunta
    if "temperatura" in question.lower():
        return "temperatura_registros"
    elif "humedad" in question.lower():
        return "humedad_registros"
    elif "calidad de aire" in question.lower():
        return "calidad_aire_registros"
    elif "sensor" in question.lower():
        return "sensores"
    elif "historial" in question.lower():
        return "historial_dispositivos"
    else:
        # Si no se encuentra una coincidencia clara, puede retornar una tabla predeterminada o lanzar un error
        return "temperatura_registros"  # Tabla predeterminada

# ==================== herramientas para bases de datos ================================#
from pydantic import BaseModel, Field
from langchain_core.tools import Tool
db_data = SQLDatabase.from_uri(uribd)
model = ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key,verbose=True)
@tool
def get_schema(question: str) -> str:
    "Herramienta para recuperar el esquema de una tabla específica,si la tabla no existe devuelve un mensaje de error controlado"
    # Usa la función 'seleccionar_tabla' para obtener el nombre de la tabla basándose en la pregunta
    tabla_seleccionada = seleccionar_tabla_v2(question)
    # Obtiene las tablas disponibles desde la base de datos
    tablas_disponibles = db_data.get_table_names()
    # Si la tabla seleccionada no existe en la base de datos, devuelve un mensaje de error controlado
    if tabla_seleccionada not in tablas_disponibles:
        return f"[ERROR] La tabla '{tabla_seleccionada}' no se reconoce. Las tablas disponibles son: {', '.join(tablas_disponibles)}"
    # Si la tabla es válida, devuelve el esquema de la tabla seleccionada
    schema = db_data.get_table_info([tabla_seleccionada])
    return schema

promptsql = ChatPromptTemplate.from_template("""
Basandonos en el esquema de tabla siguiente, escribe una consulta SQL que responda a la pregunta del usuario:
    Tabla: {table}

    Esquema: {schema}

    Pregunta: {question}
    Sql Query:
""")

sqlchain = (
    promptsql
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

@tool
def generar_sql(question: str) -> str:
    """Genera una consulta SQL a partir del esquema, la pregunta y  usando la tabla adecuada seleccionada dinámicamente """
    # Selecciona la tabla basándose en la pregunta
    tabla_seleccionada = seleccionar_tabla(question)
    schema = db_data.get_table_info([tabla_seleccionada])  # Solo obtiene el esquema de la tabla seleccionada
    return sqlchain.invoke({"schema": schema, "question": question,"table": tabla_seleccionada})

@tool
def run_query(query) -> str:
    """Herramienta que ejecuta una consulta SQL en la base de datos"""
    resultado = db_data.run(query)
    if not resultado or resultado == [(None,)]:
        return "[SIN RESULTADOS] No se encontraron registros que coincidan con la consulta."
    return resultado

promptsqlquery = ChatPromptTemplate.from_template(
    """
    Basandonos en el esquema de tabla inferior, pregunta, SQL Query y Respuesta, escribe una respuesta en lenguaje natural:
    Tabla: {table}

    Esquema: {schema}

    Pregunta: {question}
    Sql Query: {sql_query}
    SQL Respuesta: {response}

    """)

sqlnatural_chain = (
     promptsqlquery
    | model
    | StrOutputParser()
)

@tool(return_direct=True)
def generar_respuesta(question: str, sql_query: str, response: str) -> str:
    """Genera una respuesta en lenguaje natural tomando el esquema, la consulta, la query,la respuesta de la ejecucion de la query y la tabla seleccionada"""
    tabla_seleccionada = seleccionar_tabla(question)
    schema = db_data.get_table_info([tabla_seleccionada])
    return sqlnatural_chain.invoke({"schema": schema, "question": question, "sql_query":sql_query, "response":response, "table": tabla_seleccionada  })



# ===================== EMBEDDINGS =====================
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# ===================== CONEXIÓN A VECTOR STORE =====================
vector_store = ElasticsearchStore(
    es_url=ES_URL,
    es_user=ES_USER,
    es_password=ES_PASSWORD,
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever_chain = vector_store.as_retriever(search_kwargs={"k": 4})

# ========= TOOL GENERAL ==========
tool_general = retriever_chain.as_tool(
    name="informacion_general",
    description=(
        "Usa esta herramienta para responder preguntas sobre los documentos disponibles: "
        "'Detalles Técnicos del sistema', 'Asistente inteligente para uso del sistema' y "
        "'Plan de escalabilidad'. Los documentos están en español y tratan sobre domótica, uso del asistente y mejora del sistema."
    )
)


# ========= ACTUALIZACION al AGENTE DE RESPUESTA con DB==========
#modelo = ChatOpenAI(model="gpt-4-0125-preview")
#modelo = ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key)
modelo = ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key,verbose=True)

memory = MemorySaver()

system_prompt = """
Eres un asistente de bases de datos y de información técnica del sistema domótico. Si recibes consultas de información sobre las bases utiliza las herramientas
para responder las preguntas. Si recibes consultas de información de documentos solo responde usando la información disponible en los documentos.
No inventes respuestas. Si no encuentras información relevante, di que no se encontró.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{messages}"),
    ]
)

# Crea el agente con la tool general
toolkit = [tool_general, get_schema,generar_sql,run_query,generar_respuesta]
agent = create_react_agent(model=modelo, tools=toolkit, prompt=prompt)

# ========= CONSULTAR (EJEMPLO) ==========
from langchain_core.messages import HumanMessage

import uuid
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
#config = {"configurable": {"thread_id": "consulta_001"}}

def ejecutar_consulta(pregunta: str) -> str:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    # Inicializar la respuesta como string vacío
    respuesta = ""
    # Ejecutar la consulta con el agente en modo streaming
    for step in agent.stream(
        {"messages": [HumanMessage(content=pregunta)]},
        config,
        stream_mode="values",
    ):
        # Obtener el contenido del último mensaje generado
        mensaje = step["messages"][-1].content
        # Acumular en la respuesta
        respuesta = mensaje
    return respuesta



### funcion flask para consulta usando las herramientas generadas ############33
@app.route('/info', methods=['GET', 'POST'])
def consulta():
    resultado_solicitud =""
    if request.method == 'POST':
        texto_consulta = request.form.get('solicitud')
        resultado_solicitud = ejecutar_consulta(texto_consulta)
    return render_template('info.html', resultado_solicitud=resultado_solicitud)

@app.route('/consulta_docs', methods=['GET'])
def consulta_rapida():
    pregunta = request.args.get('q')
    if not pregunta:
        return jsonify({"error": "Falta el parámetro 'q' con la consulta"}), 400
    respuesta = ejecutar_consulta(pregunta)
    return jsonify({"respuesta": respuesta})
# uso es http://localhost:5000/consulta_rapida?q=¿Qué incluye el plan de escalabilidad?


#necesario si se ejecuta de manera local
#if __name__ == "__main__":
#    app.run(debug=True)