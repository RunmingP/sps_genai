import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from bigram_model import BigramModel

nlp = spacy.load("en_core_web_md") 

app = FastAPI()

# Example training corpus
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "It tells the story of Edmond Dantès who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

# Initialize the model
bigram_model = BigramModel(corpus)

# Define request schema
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):   # <-- new
    text: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embed")   # <-- new endpoint
def embed_text(request: EmbeddingRequest):
    doc = nlp(request.text)
    return {"embedding": doc.vector.tolist()}
