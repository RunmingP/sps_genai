import io
import base64
import spacy
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr
from typing import List

try:
    from app.bigram_model import BigramModel
    from app.text_rnn import Vocab, LSTMTextGenerator, train_lstm_language_model
except ModuleNotFoundError:
    from bigram_model import BigramModel
    from text_rnn import Vocab, LSTMTextGenerator, train_lstm_language_model

from helper_lib import (
    get_cifar10_loaders,
    get_data_loader,
    get_model,
    save_model,
    load_model,
    train_gan,
    cifar10_classes,
)

app = FastAPI(title="sps-genai", version="0.3.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

#spaCy
try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    raise RuntimeError(f"Failed to load spaCy model en_core_web_md: {e}")

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "It tells the story of Edmond Dantès who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)

#RNN/LSTM text generator
vocab = Vocab(corpus)
all_tokens: List[int] = []
for s in corpus:
    all_tokens += vocab.encode(s.strip().split())
lstm_device = "cpu"
lstm_model = LSTMTextGenerator(vocab_size=vocab.size, emb_dim=64, hidden=128, num_layers=1)
train_lstm_language_model(
    lstm_model, all_tokens,
    epochs=5, lr=1e-2, batch_size=64, context=4, device=lstm_device
)

#CIFAR-10
DEVICE = "cpu" 
CHECKPOINT_PATH = "/code/checkpoints/cifar10_cnn64.pt"

classifier = get_model("cnn64")
classifier.to(DEVICE)
try:
    load_model(classifier, CHECKPOINT_PATH, map_location=DEVICE)
except Exception:
    pass

#DCGAN for MNIST
GAN_CHECKPOINT = "/code/checkpoints/gan_mnist.pt"
gan_model = get_model("gan").to("cpu")
try:
    load_model(gan_model, GAN_CHECKPOINT, map_location="cpu")
except Exception:
    pass


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


@app.post("/generate_with_rnn", response_model=GenerationResponse)
def generate_with_rnn(request: TextGenerationRequest):
    words = request.start_word.strip().split()
    if not words:
        raise HTTPException(status_code=400, detail="start_word cannot be empty.")
    start_ids = vocab.encode(words)  # OOV -> <unk>
    out_ids = lstm_model.generate(start_ids, length=request.length, temperature=1.0)
    out_words = vocab.decode(out_ids)
    return {"generated_text": " ".join(out_words)}


@app.post("/embed", response_model=EmbeddingResponse)
def embed_text(request: EmbeddingRequest):
    doc = nlp(request.text)
    return {"embedding": doc.vector.tolist()}


def _preprocess_image_to_tensor(img: Image.Image, img_size: int = 64):
    from torchvision import transforms
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tf(img).unsqueeze(0)  

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
    model = classifier  
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

    save_model(model, CHECKPOINT_PATH)

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


#Endpoints
@app.post("/train_gan_mnist")
def train_gan_mnist(
    epochs: int = Query(5, ge=1, le=100),
    batch_size: int = Query(128, ge=32, le=512),
    lr: float = Query(2e-4, gt=0),
):
    """
    Train DCGAN on MNIST (images normalized to [-1,1]) and save checkpoint.
    Uses BCEWithLogitsLoss internally (see helper_lib.trainer.train_gan).
    """
    train_loader = get_data_loader("/code/data/mnist", batch_size=batch_size, train=True)

    model = gan_model  
    train_gan(
        model,
        data_loader=train_loader,
        device="cpu",
        epochs=epochs,
        z_dim=getattr(model, "z_dim", 100),
        lr=lr,
        betas=(0.5, 0.999),
        log_interval=100,
    )
    save_model(model, GAN_CHECKPOINT)
    return {"status": "trained", "epochs": epochs}


@app.get("/gan_sample")
def gan_sample(n: int = Query(16, ge=1, le=64)):
    """
    Generate n samples from trained DCGAN and return a PNG (base64).
    """
    model = gan_model
    model.eval()
    with torch.no_grad():
        z_dim = getattr(model, "z_dim", 100)
        z = torch.randn(n, z_dim)
        imgs = model.generator(z).cpu()  

        grid = vutils.make_grid(
            imgs, nrow=int(n ** 0.5), normalize=True, value_range=(-1, 1)
        )  

        np_img = (grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy())
        pil_img = Image.fromarray(np_img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return JSONResponse({"image_base64_png": b64})
