"""
Workflows del chatbot usando LangGraph.
Orquesta la ejecución de nodos mediante un StateGraph.
"""
from typing import AsyncIterator, List, Any, TypedDict, Literal, Optional
import asyncio

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from server.shared.logger import get_logger
from server.shared.env import env
from server.shared.files import read_text_file
from server.shared.database import database

from server.chatbot.nodes import (
    RouterNode,
    OffsideNode,
    ChatNode,
    QueryBuilderNode,
    QueryExecutorNode,
    SQLInterpreterNode
)

logger = get_logger(__name__)


# ============================================================================
# Definición del estado del grafo
# ============================================================================

class ChatbotState(TypedDict):
    """Estado del workflow del chatbot."""
    user_input: str
    messages: List[Any]
    route: str
    sql_query: str
    query_results: str
    response_chunks: List[str]  # Para almacenar chunks de streaming
    error: Optional[str]

# ============================================================================
# Instanciación de modelos LLM
# ============================================================================

router_llm = ChatOllama(
    model=env['ROUTER_LLM_MODEL_NAME'],
    temperature=float(env.get('ROUTER_TEMPERATURE', '0.0')),
    base_url=env['OLLAMA_BASE_URL'],
)

chat_llm = ChatOllama(
    model=env['CHAT_LLM_MODEL_NAME'],
    temperature=float(env.get('CHAT_TEMPERATURE', '0.7')),
    base_url=env['OLLAMA_BASE_URL'],
)

query_builder_llm = ChatOllama(
    model=env['QUERY_BUILDER_LLM_MODEL_NAME'],
    temperature=float(env.get('QUERY_BUILDER_TEMPERATURE', '0.0')),
    base_url=env['OLLAMA_BASE_URL'],
)

# ============================================================================
# Carga de prompts del sistema
# ============================================================================

router_system_prompt = read_text_file(env['ROUTER_NODE_SYSTEM_PROMPT_PATH'])
query_builder_system_prompt = read_text_file(env['QUERY_BUILDER_NODE_SYSTEM_PROMPT_PATH'])
query_builder_prompt_template = read_text_file(env['QUERY_BUILDER_PROMPT_TEMPLATE_PATH'])
chat_node_system_prompt = read_text_file(env['CHAT_NODE_SYSTEM_PROMPT_PATH'])
sql_interpreter_prompt_template = read_text_file(env['SQL_INTERPRETER_PROMPT_TEMPLATE_PATH'])
global_system_prompt = read_text_file(env['GLOBAL_SYSTEM_PROMPT_PATH'])

# ============================================================================
# Instanciación de nodos
# ============================================================================

router_node = RouterNode(
    llm=router_llm,
    system_prompt=router_system_prompt
)

offside_node = OffsideNode()

chat_node = ChatNode(
    llm=chat_llm,
    system_prompt=chat_node_system_prompt
)

query_builder_node = QueryBuilderNode(
    llm=query_builder_llm,
    db_url=database.db_url,
    system_prompt=query_builder_system_prompt,
    prompt_template=query_builder_prompt_template
)

query_executor_node = QueryExecutorNode(db_url=database.db_url)

sql_interpreter_node = SQLInterpreterNode(
    llm=chat_llm,
    prompt_template=sql_interpreter_prompt_template,
    system_message=SystemMessage(content=global_system_prompt)
)

# ============================================================================
# Funciones de nodos para el grafo
# ============================================================================

async def route_input(state: ChatbotState) -> ChatbotState:
    """Nodo que clasifica la entrada del usuario."""
    try:
        route = await router_node.run(state['user_input'])
        logger.info(f'Route determined: {route}')
        state['route'] = route
    except Exception as e:
        logger.error(f'Error in route_input: {e}')
        state['route'] = 'chat'
        state['error'] = str(e)
    return state


async def handle_offside(state: ChatbotState) -> ChatbotState:
    """Nodo que maneja preguntas fuera del dominio."""
    try:
        response_chunks = []
        async for chunk in offside_node.run_stream():
            response_chunks.append(chunk)

        full_response = ''.join(response_chunks)
        state['response_chunks'] = response_chunks
        state['messages'].append(HumanMessage(content=state['user_input']))
        state['messages'].append(AIMessage(content=full_response))
    except Exception as e:
        logger.error(f'Error in handle_offside: {e}')
        state['error'] = str(e)
        state['response_chunks'] = [f'Error: {str(e)}']
    return state


async def handle_chat(state: ChatbotState) -> ChatbotState:
    """Nodo que maneja conversación general."""
    try:
        response_chunks = []
        async for chunk in chat_node.run_stream(state['user_input'], state['messages']):
            response_chunks.append(chunk)

        state['response_chunks'] = response_chunks
    except Exception as e:
        logger.error(f'Error in handle_chat: {e}')
        state['error'] = str(e)
        state['response_chunks'] = [f'Sorry, an error occurred: {str(e)}']
    return state


async def build_sql_query(state: ChatbotState) -> ChatbotState:
    """Nodo que construye query SQL."""
    try:
        state['messages'].append(HumanMessage(content=state['user_input']))
        sql_query = await query_builder_node.run(state['user_input'])
        logger.info(f'SQL query built: {sql_query}')
        state['sql_query'] = sql_query
    except Exception as e:
        logger.error(f'Error in build_sql_query: {e}')
        state['error'] = str(e)
    return state


async def execute_sql_query(state: ChatbotState) -> ChatbotState:
    """Nodo que ejecuta query SQL."""
    try:
        query_results = await query_executor_node.run(state['sql_query'])
        logger.info(f'Query executed successfully')
        state['query_results'] = query_results
    except Exception as e:
        logger.error(f'Error in execute_sql_query: {e}')
        state['error'] = str(e)
    return state


async def interpret_sql_results(state: ChatbotState) -> ChatbotState:
    """Nodo que interpreta resultados SQL."""
    try:
        response_chunks = []
        async for chunk in sql_interpreter_node.run_stream(
            user_input=state['user_input'],
            sql_query=state['sql_query'],
            query_results=state['query_results']
        ):
            response_chunks.append(chunk)

        full_response = ''.join(response_chunks)
        state['response_chunks'] = response_chunks
        state['messages'].append(AIMessage(content=full_response))
    except Exception as e:
        logger.error(f'Error in interpret_sql_results: {e}')
        state['error'] = str(e)
        error_message = f'Sorry, an error occurred while interpreting results: {str(e)}'
        state['response_chunks'] = [error_message]
        state['messages'].append(AIMessage(content=error_message))
    return state


def route_by_classification(state: ChatbotState) -> Literal["offside", "chat", "sql"]:
    """Función de enrutamiento condicional basada en la clasificación."""
    return state['route']


# ============================================================================
# Construcción del grafo
# ============================================================================

def build_chatbot_graph() -> StateGraph:
    """Construye el grafo de estado del chatbot."""

    # Crear el grafo
    workflow = StateGraph(ChatbotState)

    # Agregar nodos
    workflow.add_node("route_input", route_input)
    workflow.add_node("handle_offside", handle_offside)
    workflow.add_node("handle_chat", handle_chat)
    workflow.add_node("build_sql_query", build_sql_query)
    workflow.add_node("execute_sql_query", execute_sql_query)
    workflow.add_node("interpret_sql_results", interpret_sql_results)

    # Definir el punto de entrada
    workflow.set_entry_point("route_input")

    # Agregar bordes condicionales desde route_input
    workflow.add_conditional_edges(
        "route_input",
        route_by_classification,
        {
            "offside": "handle_offside",
            "chat": "handle_chat",
            "sql": "build_sql_query"
        }
    )

    # Agregar bordes para el flujo SQL
    workflow.add_edge("build_sql_query", "execute_sql_query")
    workflow.add_edge("execute_sql_query", "interpret_sql_results")

    # Todos los nodos finales van a END
    workflow.add_edge("handle_offside", END)
    workflow.add_edge("handle_chat", END)
    workflow.add_edge("interpret_sql_results", END)

    return workflow.compile()


# Compilar el grafo una vez al inicio
chatbot_graph = build_chatbot_graph()


# ============================================================================
# Workflow principal
# ============================================================================

async def chatbot_workflow(
    user_input: str,
    messages: List[Any]
) -> AsyncIterator[str]:
    """
    Workflow principal del chatbot usando LangGraph.

    Procesa el mensaje del usuario determinando la ruta apropiada:
    - offside: Preguntas fuera del dominio
    - chat: Conversación general
    - sql: Consultas a la base de datos

    Args:
        user_input: Pregunta del usuario
        messages: Historial de mensajes de la sesión

    Yields:
        str: Fragmentos de la respuesta del chatbot
    """
    try:
        # Crear el estado inicial
        initial_state: ChatbotState = {
            'user_input': user_input,
            'messages': messages,
            'route': '',
            'sql_query': '',
            'query_results': '',
            'response_chunks': [],
            'error': None
        }

        # Ejecutar el grafo
        final_state = await chatbot_graph.ainvoke(initial_state)

        # Retornar la respuesta en streaming
        if final_state.get('error'):
            logger.error(f"Error in workflow: {final_state['error']}")

        # Hacer streaming de los chunks de respuesta
        for chunk in final_state.get('response_chunks', []):
            yield chunk

    except Exception as e:
        logger.error(f'Error in chatbot workflow: {str(e)}')
        yield f'Sorry, an error occurred while processing your message: {str(e)}'
