import io
import spacy
import torch
import torch.nn.functional as F
from PIL import Image

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr
from typing import List

# tolerant import for local run or package layout
try:
    from app.bigram_model import BigramModel
except ModuleNotFoundError:
    from bigram_model import BigramModel

# NEW: import helper_lib pieces for classifier
from helper_lib import (
    get_cifar10_loaders, get_model, save_model, load_model,
    cifar10_classes
)

app = FastAPI(title="sps-genai", version="0.2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ---------- spaCy ----------
try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    raise RuntimeError(f"Failed to load spaCy model en_core_web_md: {e}")

# ---------- Bigram demo ----------
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "It tells the story of Edmond Dantès who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: constr(strip_whitespace=True, min_length=1)
    length: int = Field(..., gt=0, le=100)

class EmbeddingRequest(BaseModel):
    text: constr(strip_whitespace=True, min_length=1)

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class GenerationResponse(BaseModel):
    generated_text: str


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse("/docs")

@app.post("/generate", response_model=GenerationResponse)
def generate_text(request: TextGenerationRequest):
    if not bigram_model.has_word(request.start_word):
        raise HTTPException(
            status_code=400,
            detail=(
                f"start_word '{request.start_word}' not found in corpus vocabulary. "
                f"Try one of: {', '.join(sorted(bigram_model.vocab_sample(10)))}"
            ),
        )
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embed", response_model=EmbeddingResponse)
def embed_text(request: EmbeddingRequest):
    doc = nlp(request.text)
    return {"embedding": doc.vector.tolist()}

# ---------- NEW: CIFAR-10 classifier in API ----------
DEVICE = "cpu"  # keep CPU for portability in grading
CHECKPOINT_PATH = "/code/checkpoints/cifar10_cnn64.pt"

# lazy init: create model and try to load a checkpoint if exists
classifier = get_model("cnn64")
classifier.to(DEVICE)
try:
    load_model(classifier, CHECKPOINT_PATH, map_location=DEVICE)
except Exception:
    # no checkpoint yet; will run with random weights until /train_cifar is called
    pass

def _preprocess_image_to_tensor(img: Image.Image, img_size: int = 64):
    from torchvision import transforms
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tf(img).unsqueeze(0)  # shape: (1,3,64,64)

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Predict CIFAR-10 class for an uploaded image.
    Returns the best class and its probability.
    """
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    x = _preprocess_image_to_tensor(img).to(DEVICE)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(x)
        probs = F.softmax(logits, dim=1)[0]
        conf, pred_idx = probs.max(dim=0)
        classes = cifar10_classes()
        pred = classes[int(pred_idx)]
        return {
            "pred_class": pred,
            "confidence": float(conf.item()),
            "top1_index": int(pred_idx.item()),
        }

@app.post("/train_cifar")
def train_cifar(epochs: int = Query(1, ge=1, le=50), batch_size: int = Query(128, ge=16, le=512)):
    """
    Optional: Train the CIFAR-10 classifier for a few epochs and save a checkpoint.
    WARNING: Training on CPU is slow; keep epochs small for demo/grading.
    """
    import torch.optim as optim
    import torch.nn as nn

    train_loader, test_loader = get_cifar10_loaders("/code/data/cifar10", batch_size=batch_size, img_size=64, augment=True)
    model = classifier  # use the global one
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        running = 0.0
        for i, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        avg_loss = running / len(train_loader.dataset)

    # save checkpoint and evaluate quick accuracy
    save_model(model, CHECKPOINT_PATH)

    # quick eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    acc = correct / total if total > 0 else 0.0

    return {"status": "trained", "epochs": epochs, "avg_train_loss": round(avg_loss, 4), "test_acc": round(acc, 4)}
