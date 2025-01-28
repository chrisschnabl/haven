FROM python:3.9-slim-bookworm

COPY main.py .
RUN pip install datasets torch transformers

CMD ["/usr/local/bin/python3", "main.py"]