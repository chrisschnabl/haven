FROM python:3.9-slim-bookworm  # Try different base images
    
COPY main.py .

RUN pip install datasets torch transformers  # Try reduce it

CMD ["/usr/local/bin/python3", "main.py"]