import io
import base64
from pathlib import Path
from typing import List, Optional

import spacy
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, constr
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

try:
    from app.text_rnn import Vocab, LSTMTextGenerator, train_lstm_language_model
except ModuleNotFoundError:
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
from helper_lib.generator import generate_samples, generate_ebm_samples

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_ROOT = BASE_DIR / "checkpoints"
DATA_ROOT = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="sps-genai", version="0.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    raise RuntimeError(f"Failed to load spaCy model: {e}")

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "It tells the story of Edmond Dantès who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
vocab = Vocab(corpus)
all_tokens = []
for s in corpus:
    all_tokens += vocab.encode(s.strip().split())

lstm_device = "cpu"
lstm_model = LSTMTextGenerator(vocab_size=vocab.size, emb_dim=64, hidden=128, num_layers=1)
train_lstm_language_model(
    lstm_model,
    all_tokens,
    epochs=5,
    lr=1e-2,
    batch_size=64,
    context=4,
    device=lstm_device,
)

GPT_BASE_MODEL = "openai-community/gpt2"
GPT_CHECKPOINT_DIR = CHECKPOINT_ROOT / "gpt2_qa"
GPT_CHECKPOINT_DIR_RL = CHECKPOINT_ROOT / "gpt2_qa_rl"
tokenizer = AutoTokenizer.from_pretrained(GPT_BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

try:
    try:
        gpt_model = AutoModelForCausalLM.from_pretrained(GPT_CHECKPOINT_DIR_RL)
    except Exception:
        gpt_model = AutoModelForCausalLM.from_pretrained(GPT_CHECKPOINT_DIR)
except Exception:
    gpt_model = AutoModelForCausalLM.from_pretrained(GPT_BASE_MODEL)
gpt_model.to("cpu")


class LLMTrainConfig(BaseModel):
    epochs: int = Field(1, ge=1, le=3)
    max_train_samples: int = Field(1024, ge=32, le=5000)
    max_eval_samples: int = Field(256, ge=0, le=2000)
    batch_size: int = Field(4, ge=1, le=16)
    lr: float = Field(5e-5, gt=0)
    max_length: int = Field(256, ge=64, le=512)


def fine_tune_gpt2_on_nectar(config: LLMTrainConfig):
    from torch.utils.data import DataLoader

    ds = load_dataset("berkeley-nest/Nectar", split="train")
    n = len(ds)
    train_n = min(config.max_train_samples, n)
    eval_n = min(config.max_eval_samples, max(0, n - train_n))

    train_ds = ds.select(range(train_n))
    eval_ds = ds.select(range(train_n, train_n + eval_n)) if eval_n > 0 else None

    def build_qa_text(example):
        prompt = example.get("prompt", "")
        answers = example.get("answers", [])
        best_answer_text = ""
        if answers:
            best = min(answers, key=lambda a: a.get("rank", 9999))
            best_answer_text = best.get("answer", "")

        text = f"Question:\n{prompt}\n\nAnswer:\n{best_answer_text}"
        tok = tokenizer(
            text,
            max_length=config.max_length,
            truncation=True,
            padding="max_length",
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    train_ds = train_ds.map(build_qa_text, remove_columns=train_ds.column_names)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    if eval_ds is not None:
        eval_ds = eval_ds.map(build_qa_text, remove_columns=eval_ds.column_names)
        eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        eval_loader = DataLoader(eval_ds, batch_size=config.batch_size)
    else:
        eval_loader = None

    device = "cpu"
    gpt_model.to(device)
    gpt_model.train()
    optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=config.lr)

    last_train_loss, last_eval_loss = None, None
    for ep in range(1, config.epochs + 1):
        total_loss, steps = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = gpt_model(**batch).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        last_train_loss = total_loss / max(1, steps)

        if eval_loader:
            gpt_model.eval()
            eval_tot, eval_steps = 0.0, 0
            with torch.no_grad():
                for batch in eval_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    eval_tot += float(gpt_model(**batch).loss.item())
                    eval_steps += 1
            last_eval_loss = eval_tot / max(1, eval_steps)
            gpt_model.train()

    GPT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    gpt_model.save_pretrained(GPT_CHECKPOINT_DIR)
    tokenizer.save_pretrained(GPT_CHECKPOINT_DIR)

    return last_train_loss, last_eval_loss


class LLMRLTrainConfig(BaseModel):
    epochs: int = Field(1, ge=1, le=3)
    max_train_samples: int = Field(64, ge=1, le=512)
    max_new_tokens: int = Field(64, ge=8, le=256)
    lr: float = Field(5e-6, gt=0)


RL_PROMPTS = [
    "What is reinforcement learning?",
    "Can you explain machine learning?",
    "What is supervised learning?",
    "What is the difference between supervised and unsupervised learning?",
    "How does a neural network work?",
    "What is overfitting in machine learning?",
    "What is a Markov decision process?",
    "What are the main components of reinforcement learning?",
]


def _compute_rl_reward(text: str) -> float:
    s = text.strip()
    ok_start = s.startswith("That is a great question")
    ok_end = s.endswith("let me know if you have any other questions")
    if ok_start and ok_end:
        return 1.0
    if ok_start or ok_end:
        return 0.5
    return -1.0


def rl_post_train_gpt2(config: LLMRLTrainConfig):
    device = "cpu"
    gpt_model.to(device)
    gpt_model.train()
    optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=config.lr)

    total_loss = 0.0
    total_steps = 0

    for ep in range(1, config.epochs + 1):
        for i in range(config.max_train_samples):
            prompt = RL_PROMPTS[i % len(RL_PROMPTS)]

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen_ids_full = gpt_model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = gen_ids_full[0, inputs["input_ids"].shape[1] :]
            generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            reward = _compute_rl_reward(generated_text)

            if reward == 0.0 or gen_ids.numel() == 0:
                continue

            all_input_ids = torch.cat(
                [inputs["input_ids"], gen_ids.unsqueeze(0)], dim=1
            )
            attention_mask = torch.ones_like(all_input_ids, device=device)

            outputs = gpt_model(all_input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = all_input_ids[:, 1:].contiguous()

            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            input_len = inputs["input_ids"].shape[1]
            cont_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
            cont_mask[:, input_len - 1 :] = True

            cont_log_probs = token_log_probs[cont_mask]
            if cont_log_probs.numel() == 0:
                continue

            loss = -reward * cont_log_probs.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_steps += 1

    avg_loss = total_loss / max(1, total_steps)

    GPT_CHECKPOINT_DIR_RL.mkdir(parents=True, exist_ok=True)
    gpt_model.save_pretrained(GPT_CHECKPOINT_DIR_RL)
    tokenizer.save_pretrained(GPT_CHECKPOINT_DIR_RL)

    return avg_loss


DEVICE = "cpu"
CHECKPOINT_PATH = CHECKPOINT_ROOT / "cifar10_cnn64.pt"

classifier = get_model("cnn64")
classifier.to(DEVICE)
try:
    load_model(classifier, CHECKPOINT_PATH, map_location=DEVICE)
except Exception:
    pass

GAN_CHECKPOINT = CHECKPOINT_ROOT / "gan_mnist.pt"
gan_model = get_model("gan").to("cpu")
try:
    load_model(gan_model, GAN_CHECKPOINT, map_location="cpu")
except Exception:
    pass

DIFFUSION_MODEL = get_model("diffusion").to("cpu")
EBM_MODEL = get_model("ebm").to("cpu")


class TextGenerationRequest(BaseModel):
    start_word: constr(strip_whitespace=True, min_length=1)
    length: int = Field(..., gt=0, le=100)


class EmbeddingRequest(BaseModel):
    text: constr(strip_whitespace=True, min_length=1)


class EmbeddingResponse(BaseModel):
    embedding: List[float]


class GenerationResponse(BaseModel):
    generated_text: str


class QARequest(BaseModel):
    question: constr(strip_whitespace=True, min_length=1)
    context: Optional[str] = None
    max_new_tokens: int = Field(64, gt=0, le=256)


class QAResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse("/docs")


@app.post("/train_llm")
def train_llm(config: LLMTrainConfig):
    train_loss, eval_loss = fine_tune_gpt2_on_nectar(config)
    return {"status": "ok", "train_loss": train_loss, "eval_loss": eval_loss}


@app.post("/train_llm_rl")
def train_llm_rl(config: LLMRLTrainConfig):
    avg_policy_loss = rl_post_train_gpt2(config)
    return {"status": "ok", "avg_policy_loss": avg_policy_loss}


@app.post("/generate", response_model=GenerationResponse)
def generate_text(request: TextGenerationRequest):
    words = request.start_word.strip().split()
    start_ids = vocab.encode(words)
    out_ids = lstm_model.generate(start_ids, length=request.length, temperature=1.0)
    out_words = vocab.decode(out_ids)
    return {"generated_text": " ".join(out_words)}


@app.post("/generate_with_llm", response_model=GenerationResponse)
def generate_with_llm(request: TextGenerationRequest):
    prompt = request.start_word.strip()
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        output_ids = gpt_model.generate(
            **inputs,
            max_new_tokens=request.length,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {"generated_text": generated}


@app.post("/embed", response_model=EmbeddingResponse)
def embed_text(request: EmbeddingRequest):
    doc = nlp(request.text)
    return {"embedding": doc.vector.tolist()}


@app.post("/qa", response_model=QAResponse)
def answer_question(request: QARequest):
    if request.context:
        prompt = f"Context:\n{request.context}\n\nQuestion: {request.question}\nAnswer:"
    else:
        prompt = f"Question: {request.question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        output_ids = gpt_model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return {"answer": answer}


def _preprocess_image_to_tensor(img: Image.Image, img_size: int = 64):
    from torchvision import transforms

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return tf(img).unsqueeze(0)


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
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
def train_cifar(
    epochs: int = Query(1, ge=1, le=50),
    batch_size: int = Query(128, ge=16, le=512),
):
    import torch.optim as optim
    import torch.nn as nn

    train_loader, test_loader = get_cifar10_loaders(
        str(DATA_ROOT / "cifar10"), batch_size=batch_size, img_size=64, augment=True
    )
    model = classifier
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        running = 0.0
        for imgs, targets in train_loader:
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
            preds = model(imgs).argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    acc = correct / total if total > 0 else 0.0

    return {
        "status": "trained",
        "epochs": epochs,
        "avg_train_loss": round(avg_loss, 4),
        "test_acc": round(acc, 4),
    }


@app.post("/train_gan_mnist")
def train_gan_mnist(
    epochs: int = Query(5, ge=1, le=100),
    batch_size: int = Query(128, ge=32, le=512),
    lr: float = Query(2e-4, gt=0),
):
    train_loader = get_data_loader(
        str(DATA_ROOT / "mnist"), batch_size=batch_size, train=True
    )
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
    model = gan_model
    model.eval()
    with torch.no_grad():
        z_dim = getattr(model, "z_dim", 100)
        z = torch.randn(n, z_dim)
        imgs = model.generator(z).cpu()

        grid = vutils.make_grid(
            imgs, nrow=int(n ** 0.5), normalize=True, value_range=(-1, 1)
        )
        np_img = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
        pil_img = Image.fromarray(np_img)

        out_path = OUTPUT_DIR / "gan_samples.png"
        pil_img.save(out_path)

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "num_samples": int(n),
        "saved_path": str(out_path),
        "image_base64_png": b64,
    }


@app.get("/diffusion_sample")
def diffusion_sample(
    n: int = Query(16, ge=1, le=64),
    steps: int = Query(100, ge=10, le=1000),
):
    imgs = generate_samples(
        DIFFUSION_MODEL, device="cpu", num_samples=n, diffusion_steps=steps
    )

    grid = vutils.make_grid(imgs, nrow=int(n ** 0.5), normalize=True)
    np_img = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
    pil_img = Image.fromarray(np_img)

    out_path = OUTPUT_DIR / "diffusion_samples.png"
    pil_img.save(out_path)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "num_samples": int(n),
        "saved_path": str(out_path),
        "image_base64_png": b64,
    }


@app.get("/ebm_sample")
def ebm_sample(
    n: int = Query(16, ge=1, le=64),
    steps: int = Query(50, ge=5, le=500),
    step_size: float = Query(0.1, gt=0, le=1.0),
    noise: bool = Query(False),
):
    imgs = generate_ebm_samples(
        model=EBM_MODEL,
        device="cpu",
        num_samples=n,
        steps=steps,
        step_size=step_size,
        add_noise=noise,
    )

    grid = vutils.make_grid(imgs, nrow=int(n ** 0.5), normalize=True)
    np_img = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
    pil_img = Image.fromarray(np_img)

    out_path = OUTPUT_DIR / "ebm_samples.png"
    pil_img.save(out_path)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "num_samples": int(n),
        "saved_path": str(out_path),
        "image_base64_png": b64,
    }
