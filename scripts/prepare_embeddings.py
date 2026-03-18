"""
Script para generar embeddings desde docs/embeddings y guardarlos en database/embeddings
"""

import os
import re
from logging import getLogger
from typing import List

import faiss
import numpy as np
import regex
from dotenv import dotenv_values
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Cargar configuración del entorno
env = dotenv_values('.env')

logger = getLogger(__name__)
logger.setLevel(env.get('LOG_LEVEL', 'INFO').upper())

# Expresión regular para URLs
URL_PATTERN = r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*"


def load_documents(docs_path: str) -> List:
    """
    Carga documentos PDF y TXT desde un directorio.
    Para PDFs, agrega "PAGE: N" como primera línea del contenido.

    Args:
        docs_path: Ruta del directorio con los archivos

    Returns:
        List: Lista de documentos cargados
    """
    documents = []

    # Cargar documentos PDF
    pdf_loader = DirectoryLoader(
        docs_path,
        glob='**/*.pdf',
        show_progress=True,
        loader_cls=PyPDFLoader,
        loader_kwargs={'mode': 'page'},
    )

    pdf_documents = pdf_loader.load()

    # Agregar número de página a cada documento PDF
    for doc in pdf_documents:
        page_num = doc.metadata.get('page', 0) + 1  # +1 porque suele empezar en 0
        doc.page_content = f"PAGE: {page_num}\n{doc.page_content}"
        documents.append(doc)

    # Cargar documentos TXT
    txt_loader = DirectoryLoader(
        docs_path,
        glob='**/*.txt',
        show_progress=True,
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
    )

    txt_documents = txt_loader.load()
    documents.extend(txt_documents)

    logger.info('Documentos cargados: %d PDFs, %d TXTs', len(pdf_documents), len(txt_documents))

    return documents


def clean_text(text: str) -> str:
    """
    Limpia y normaliza el texto de un documento.
    Preserva "PAGE: N" si está al inicio.

    Args:
        text: Texto a limpiar

    Returns:
        str: Texto limpio y normalizado
    """
    # Detectar si comienza con PAGE:
    page_prefix = ''
    if text.startswith('PAGE: '):
        lines = text.split('\n', 1)
        page_prefix = lines[0] + '\n'
        text = lines[1] if len(lines) > 1 else ''

    # Eliminar saltos de línea múltiples
    text = re.sub(r'\n+', ' ', text)
    # Eliminar URLs
    text = re.sub(URL_PATTERN, '', text)
    # Eliminar caracteres especiales
    text = re.sub(r'(?:-{2,}|[^\w\s-]|_)+', ' ', text)
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text)
    # Eliminar espacios al principio y al final
    text = text.strip()

    return page_prefix + text


def filter_documents(raw_documents: List, chunk_overlap: int) -> List:
    """
    Filtra documentos basándose en longitud y cantidad de letras.

    Args:
        raw_documents: Lista de documentos sin filtrar
        chunk_overlap: Tamaño mínimo del chunk para filtrado

    Returns:
        List: Lista de documentos filtrados
    """
    filtered_documents = []

    for doc in raw_documents:
        text = clean_text(doc.page_content)
        letter_quantity = len(regex.findall(r'\p{L}', text))

        if letter_quantity <= 200 or len(text) <= chunk_overlap:
            logger.debug('Documento filtrado por longitud insuficiente')
            continue

        doc.page_content = text
        filtered_documents.append(doc)

    logger.info('Documentos después del filtrado: %d de %d', len(filtered_documents), len(raw_documents))

    return filtered_documents


def split_documents(filtered_documents: List, chunk_size: int, chunk_overlap: int) -> List:
    """
    Divide documentos en fragmentos más pequeños.

    Args:
        filtered_documents: Lista de documentos filtrados
        chunk_size: Tamaño de cada fragmento
        chunk_overlap: Superposición entre fragmentos

    Returns:
        List: Lista de fragmentos de documentos
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    documents = text_splitter.split_documents(filtered_documents)

    logger.info('Fragmentos generados: %d', len(documents))

    return documents


def create_faiss_index(documents: List, embeddings: OllamaEmbeddings, output_path: str) -> None:
    """
    Crea y guarda un índice FAISS con embeddings normalizados L2.
    Usa Inner Product para emular cosine similarity.

    Args:
        documents: Lista de documentos fragmentados
        embeddings: Modelo de embeddings de Ollama
        output_path: Ruta donde guardar el índice FAISS
    """
    logger.info('Generando embeddings para %d fragmentos...', len(documents))

    # Extraer textos de los documentos y validar
    valid_documents = []
    texts = []

    for doc in documents:
        text = doc.page_content.strip()
        # Validar que el texto tenga contenido significativo
        if len(text) > 10 and len(text.replace(' ', '')) > 5:
            texts.append(text)
            valid_documents.append(doc)
        else:
            logger.debug('Documento descartado por contenido insuficiente: %s', text[:50])

    if not texts:
        logger.error('No hay textos válidos para generar embeddings')
        return

    logger.info('Textos válidos para embeddings: %d de %d', len(texts), len(documents))

    # Generar embeddings con manejo de errores
    try:
        embeddings_list = embeddings.embed_documents(texts)
    except Exception as e:
        logger.error('Error al generar embeddings: %s', str(e))
        logger.info('Intentando generar embeddings uno por uno...')

        embeddings_list = []
        final_documents = []

        for i, text in enumerate(texts):
            try:
                embedding = embeddings.embed_documents([text])
                # Verificar que no contenga NaN o Inf
                if embedding and not any(np.isnan(embedding[0])) and not any(np.isinf(embedding[0])):
                    embeddings_list.append(embedding[0])
                    final_documents.append(valid_documents[i])
                else:
                    logger.warning('Embedding inválido para chunk %d (NaN o Inf detectado)', i)
            except Exception as chunk_error:
                logger.warning('Error al procesar chunk %d: %s', i, str(chunk_error))
                continue

        valid_documents = final_documents
        logger.info('Embeddings generados exitosamente: %d', len(embeddings_list))

    if not embeddings_list:
        logger.error('No se pudo generar ningún embedding válido')
        return


    if not embeddings_list:
        logger.error('No se pudo generar ningún embedding válido')
        return

    # Convertir a numpy array float32
    embeddings_array = np.array(embeddings_list, dtype=np.float32)

    # Verificar que no haya NaN o Inf en los embeddings
    if np.isnan(embeddings_array).any():
        logger.error('Se detectaron valores NaN en los embeddings')
        return

    if np.isinf(embeddings_array).any():
        logger.error('Se detectaron valores Inf en los embeddings')
        return

    # Normalizar L2 (clave para emular cosine con Inner Product)
    faiss.normalize_L2(embeddings_array)

    # Crear índice FAISS con Inner Product
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Agregar embeddings al índice
    index.add(embeddings_array)

    logger.info('Índice FAISS creado con %d vectores de dimensión %d', index.ntotal, dimension)

    # Guardar índice
    index_path = os.path.join(output_path, 'index.faiss')
    faiss.write_index(index, index_path)
    logger.info('Índice guardado en: %s', index_path)

    # Guardar metadatos de documentos
    metadata = {
        'documents': [
            {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'index': idx
            }
            for idx, doc in enumerate(valid_documents)
        ],
        'dimension': dimension,
        'total_vectors': index.ntotal,
    }

    import pickle
    metadata_path = os.path.join(output_path, 'index.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info('Metadatos guardados en: %s', metadata_path)


def main() -> None:
    """
    Función principal que orquesta el proceso de generación de embeddings.
    """
    # Configuración desde variables de entorno
    docs_path = env.get('DOCS_EMBEDDINGS_DIR_PATH')
    output_path = env.get('DATABASE_EMBEDDINGS_DIR_PATH')
    embeddings_model_name = env.get('EMBEDDINGS_MODEL_NAME')
    chunk_size = env.get('CHUNK_SIZE')
    chunk_overlap = env.get('CHUNK_OVERLAP')

    if not docs_path \
        or not output_path \
        or not embeddings_model_name \
        or not chunk_size \
        or not chunk_overlap:
        logger.error('Faltan variables de entorno requeridas. Verifique .env')
        return

    chunk_size = int(chunk_size)
    chunk_overlap = int(chunk_overlap)

    # Validar que existen las variables requeridas
    if not docs_path:
        logger.error('Variable de entorno DOCS_EMBEDDINGS_DIR_PATH no está definida')
        return

    if not output_path:
        logger.error('Variable de entorno DATABASE_EMBEDDINGS_DIR_PATH no está definida')
        return

    logger.info('=== Iniciando generación de embeddings ===')
    logger.info('Documentos origen: %s', docs_path)
    logger.info('Salida: %s', output_path)
    logger.info('Modelo: %s', embeddings_model_name)
    logger.info('Chunk size: %d, overlap: %d', chunk_size, chunk_overlap)

    # Verificar que existe el directorio de documentos
    if not os.path.exists(docs_path):
        logger.error('El directorio de documentos no existe: %s', docs_path)
        return

    # Verificar que existe el directorio de salida
    if not os.path.exists(output_path):
        logger.error('El directorio de salida no existe: %s', output_path)
        return

    # Cargar documentos
    raw_documents = load_documents(docs_path)

    if len(raw_documents) == 0:
        logger.warning('No se encontraron documentos en %s', docs_path)
        return

    # Filtrar documentos
    filtered_documents = filter_documents(raw_documents, chunk_overlap)

    if len(filtered_documents) == 0:
        logger.warning('No hay documentos válidos después del filtrado')
        return

    # Dividir documentos
    documents = split_documents(filtered_documents, chunk_size, chunk_overlap)

    if len(documents) == 0:
        logger.warning('No se generaron fragmentos')
        return

    # Crear modelo de embeddings
    logger.info('Inicializando modelo de embeddings...')
    embeddings = OllamaEmbeddings(model=embeddings_model_name)

    # Crear índice FAISS
    create_faiss_index(documents, embeddings, output_path)

    logger.info('=== Proceso completado exitosamente ===')


if __name__ == '__main__':
    main()
