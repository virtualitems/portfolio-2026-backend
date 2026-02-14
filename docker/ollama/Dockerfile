FROM ollama/ollama:latest

RUN ollama serve & \
    sleep 10 && \
    echo "Descargando modelo mistral:7b..." && \
    ollama pull mistral:7b && \
    echo "Descargando modelo qwen2.5-coder:7b..." && \
    ollama pull qwen2.5-coder:7b && \
    echo "Descargando modelo llama3.1:8b..." && \
    ollama pull llama3.1:8b && \
    echo "Modelos descargados exitosamente" && \
    pkill ollama

EXPOSE 11434

CMD ["ollama", "serve"]