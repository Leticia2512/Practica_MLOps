from fastapi import FastAPI
from transformers import pipeline

app = FastAPI(
    title="API de despliegue MLOps",
    description="API REST con FastAPI para la práctica final. Incluye endpoints de operaciones básicas y modelos de Hugging Face.",
    version="1.0"
)

# Hugging Face pipelines
sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@app.get("/")
def root():
    return {"message": "API de ejemplo para la práctica MLOps"}

@app.get("/sentiment/")
def get_sentiment(text: str):
    """Analiza el sentimiento de un texto usando Hugging Face."""
    result = sentiment_analyzer(text)
    return {"input": text, "sentiment": result}

@app.get("/summarize/")
def get_summary(text: str):
    """Resume un texto usando Hugging Face."""
    summary = summarizer(text, max_length=40, min_length=10, do_sample=False)
    return {"input": text, "summary": summary[0]['summary_text']}

@app.get("/hello/")
def hello(name: str = "Mundo"):
    return {"message": f"Hola, {name}!"}

@app.get("/add/")
def add(a: float, b: float):
    return {"a": a, "b": b, "sum": a + b}

@app.get("/multiply/")
def multiply(a: float, b: float):
    return {"a": a, "b": b, "product": a * b}
