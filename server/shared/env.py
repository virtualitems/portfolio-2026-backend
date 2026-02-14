import os
from dotenv import dotenv_values

env = dotenv_values('.env')

required_vars = [
    'DB_NAME',
    'DB_HOST',
    'DB_PASSWORD',
    'DB_PORT',
    'DB_USER',
    'BASE_URL',
    'MEDIA_FILES_DIR_PATH',
    'STATIC_FILES_DIR_PATH',
    'YOLO_SAFETY_DETECTOR_MODEL_PATH',
    'GLOBAL_SYSTEM_PROMPT_PATH',
    'ROUTER_LLM_MODEL_NAME',
    'ROUTER_NODE_SYSTEM_PROMPT_PATH',
    'ROUTER_TEMPERATURE',
    'CHAT_LLM_MODEL_NAME',
    'CHAT_NODE_SYSTEM_PROMPT_PATH',
    'CHAT_TEMPERATURE',
    'QUERY_BUILDER_LLM_MODEL_NAME',
    'QUERY_BUILDER_NODE_SYSTEM_PROMPT_PATH',
    'QUERY_BUILDER_PROMPT_TEMPLATE_PATH',
    'QUERY_BUILDER_TEMPERATURE',
    'QUERY_EXECUTOR_LLM_MODEL_NAME',
    'QUERY_EXECUTOR_NODE_SYSTEM_PROMPT_PATH',
    'QUERY_EXECUTOR_TEMPERATURE',
    'SQL_INTERPRETER_PROMPT_TEMPLATE_PATH',
    'OLLAMA_BASE_URL',
    'LOG_LEVEL'
]

path_vars = [
    'MEDIA_FILES_DIR_PATH',
    'STATIC_FILES_DIR_PATH',
    'YOLO_SAFETY_DETECTOR_MODEL_PATH',
    'GLOBAL_SYSTEM_PROMPT_PATH',
    'ROUTER_NODE_SYSTEM_PROMPT_PATH',
    'CHAT_NODE_SYSTEM_PROMPT_PATH',
    'QUERY_BUILDER_NODE_SYSTEM_PROMPT_PATH',
    'QUERY_BUILDER_PROMPT_TEMPLATE_PATH',
    'QUERY_EXECUTOR_NODE_SYSTEM_PROMPT_PATH',
    'SQL_INTERPRETER_PROMPT_TEMPLATE_PATH'
]

numeric_vars = [
    'ROUTER_TEMPERATURE',
    'CHAT_TEMPERATURE',
    'QUERY_BUILDER_TEMPERATURE',
    'QUERY_EXECUTOR_TEMPERATURE'
]

valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

if env['LOG_LEVEL'].upper() not in valid_log_levels:
    log_levels = ', '.join(valid_log_levels)
    received_level = env['LOG_LEVEL']
    raise ValueError(f'LOG_LEVEL debe ser uno de {log_levels}, se recibió: {received_level}')

for var in required_vars:
    if var not in env:
        raise ValueError(f'Falta la variable de entorno requerida: {var}')
    if not env[var].strip():
        raise ValueError(f'La variable de entorno no puede estar vacía: {env[var]}')

for var in path_vars:
    if not os.path.exists(env[var]):
        raise ValueError(f'La ruta no existe: {env[var]}')

for var in numeric_vars:
    try:
        float(env[var])
    except ValueError:
        raise ValueError(f'No es un número válido: {env[var]}')
