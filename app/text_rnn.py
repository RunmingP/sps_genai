import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class Vocab:
    """Word-level vocabulary with <pad>=0 and <unk>=1."""

    def __init__(self, corpus: List[str]):
        tokens = []
        for s in corpus:
            tokens.extend(s.strip().split())
        uniq = sorted(set(tokens))
        self.stoi: Dict[str, int] = {w: i + 2 for i, w in enumerate(uniq)}
        self.stoi["<pad>"] = 0
        self.stoi["<unk>"] = 1
        self.itos: List[str] = [None] * len(self.stoi)
        for w, i in self.stoi.items():
            self.itos[i] = w

    def encode(self, words: List[str]) -> List[int]:
        return [self.stoi.get(w, 1) for w in words]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids]

    @property
    def size(self) -> int:
        return len(self.stoi)


class LSTMTextGenerator(nn.Module):
    """Lightweight word-level LSTM: Embedding → LSTM → Linear."""

    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden: int = 128, num_layers: int = 1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        x = self.emb(x)
        out, h = self.lstm(x, h)
        logits = self.proj(out)
        return logits, h

    @torch.no_grad()
    def generate(self, start_ids: List[int], length: int, temperature: float = 1.0) -> List[int]:
        self.eval()
        device = next(self.parameters()).device
        ids = start_ids[:]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _, h = self.forward(x)

        cur = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
        for _ in range(max(0, length - len(ids))):
            logits, h = self.forward(cur, h)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids.append(int(next_id.item()))
            cur = next_id
        return ids


def make_lm_pairs(seq: List[int], context: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (X, y) pairs: up to `context` previous tokens predict the current token."""
    X, y = [], []
    for i in range(1, len(seq)):
        left = max(0, i - context)
        X.append(seq[left:i])
        y.append(seq[i])
    maxlen = max(len(s) for s in X) if X else 1
    for i in range(len(X)):
        X[i] = [0] * (maxlen - len(X[i])) + X[i]  
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def train_lstm_language_model(
    model: LSTMTextGenerator,
    token_ids: List[int],
    epochs: int = 5,
    lr: float = 1e-2,
    batch_size: int = 64,
    context: int = 4,
    device: str = "cpu",
):
    """Minimal training loop with cross-entropy on the last timestep."""
    model.to(device)
    X, y = make_lm_pairs(token_ids, context=context)
    if len(X) == 0:
        return model

    ds = torch.utils.data.TensorDataset(X, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs + 1):
        run, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = loss_fn(logits[:, -1, :], yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            run += loss.item() * xb.size(0)
            n += xb.size(0)
        print(f"[LSTM LM] epoch {ep}: loss={run / max(1, n):.4f}")
    return model
