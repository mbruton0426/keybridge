"""The model (same architecture as in the paper), training, prediction and saving."""
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .common import EOS, PAD, SOS, say, stop
from .data import collate, encode

FORMAT_NAME = "shared-space-decipherment"
FORMAT_VERSION = 1

DEFAULT_HYPERPARAMETERS = {
    "embedding_dim": 512, "hidden_size": 320, "num_layers": 3, "num_heads": 2,
    "dropout": 0.44, "learning_rate": 4e-4, "batch_size": 16,
}


class DecryptionLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_size, num_layers,
                 num_heads, dropout, pad_id=0, **_):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(hidden_size * 2, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 4)
        self.fc = nn.Linear(hidden_size * 4, output_size)
        for name, p in self.named_parameters():
            if "weight" in name:
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p, -0.1, 0.1)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x):
        mask = x == self.pad_id
        h, _ = self.lstm(self.embedding(x))
        h = self.dropout(h)
        ctx, _ = self.attn(h, h, h, key_padding_mask=mask)
        return self.fc(self.layer_norm(torch.cat((h, ctx), dim=2)))


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _content_mask(y, out_vocab):
    return (y != out_vocab[PAD]) & (y != out_vocab[SOS]) & (y != out_vocab[EOS])


@torch.no_grad()
def _validate(model, dl, loss_fn, out_vocab, dev, use_amp):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        with torch.autocast(device_type=dev, enabled=use_amp):
            out = model(x)
            total_loss += loss_fn(out.reshape(-1, out.size(-1)), y.reshape(-1)).item()
        m = _content_mask(y, out_vocab)
        correct += (out.argmax(-1)[m] == y[m]).sum().item()
        total += m.sum().item()
    return total_loss / max(1, len(dl)), correct / max(1, total)


def train_model(train_ds, val_ds, in_vocab, out_vocab, hp, max_epochs=100, patience=10,
                seed=9, log=print):
    """Trains until validation loss stops improving for `patience` rounds, or validation
    accuracy has been 100% for 5 rounds. Returns the best model, history and seconds."""
    torch.manual_seed(seed)
    dev = device()
    model = DecryptionLSTM(len(in_vocab), len(out_vocab), pad_id=in_vocab[PAD], **hp).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=out_vocab[PAD])
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True, collate_fn=collate, generator=g)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)

    best, best_loss, bad, perfect, history = None, float("inf"), 0, 0, []
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev, enabled=use_amp):
                out = model(x)
                loss = loss_fn(out.reshape(-1, out.size(-1)), y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        val_loss, val_acc = _validate(model, val_dl, loss_fn, out_vocab, dev, use_amp)
        sched.step(val_loss)
        history.append({"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_acc})
        improved = val_loss < best_loss
        if improved:
            best_loss, bad = val_loss, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        say(log, f"  round {epoch:3d}   checking accuracy: {val_acc:6.1%}" + ("   (best so far)" if improved else ""))
        if bad >= patience:
            say(log, f"  No improvement for {patience} rounds, so training stopped.")
            break
        perfect = perfect + 1 if val_acc >= 0.9999 else 0
        if perfect >= 5:
            say(log, "  100% on the checking examples for 5 rounds, so training stopped.")
            break
    model.load_state_dict(best)
    return model, history, time.time() - start


def tune(train_ds, val_ds, in_vocab, out_vocab, n_trials, seed=9, epochs=10, log=print):
    try:
        import optuna
    except ImportError:
        stop("The model-settings search needs the 'optuna' package: pip install optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        hp = {
            "embedding_dim": trial.suggest_int("embedding_dim", 64, 512, step=64),
            "hidden_size": trial.suggest_int("hidden_size", 64, 512, step=64),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.2, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
            "batch_size": trial.suggest_int("batch_size", 16, 128, step=16),
            "num_heads": trial.suggest_categorical("num_heads", [1, 2, 4, 8]),
        }
        model, hist, _ = train_model(train_ds, val_ds, in_vocab, out_vocab, hp, max_epochs=epochs,
                                     patience=epochs, seed=seed, log=None)
        acc = max(h["val_accuracy"] for h in hist)
        say(log, f"  trial {trial.number + 1}/{n_trials}: accuracy {acc:.1%}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return acc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials,
                   callbacks=[lambda s, t: s.stop() if s.best_value >= 0.995 else None])
    return dict(study.best_params), study.best_value


@torch.no_grad()
def predict(model, symbol_lines, in_vocab, out_vocab, batch_size=64):
    """For each line, a list of (predicted unit, confidence) per cipher symbol."""
    model.eval()
    dev = next(model.parameters()).device
    id_to_unit = {i: u for u, i in out_vocab.items()}
    results = []
    for start in range(0, len(symbol_lines), batch_size):
        batch = symbol_lines[start:start + batch_size]
        encoded = [encode(s, in_vocab) for s in batch]
        x = torch.zeros(len(batch), max(len(e) for e in encoded), dtype=torch.long)
        for i, e in enumerate(encoded):
            x[i, :len(e)] = torch.tensor(e)
        conf, idx = torch.softmax(model(x.to(dev)).float(), dim=-1).max(-1)
        for i, line in enumerate(batch):
            results.append([(id_to_unit[idx[i, t].item()], conf[i, t].item()) for t in range(1, len(line) + 1)])
    return results


def score(predictions, gold_units):
    """Accuracy and support-weighted F1 over content positions (as in the paper)."""
    from sklearn.metrics import f1_score
    y_true, y_pred = [], []
    for pred, gold in zip(predictions, gold_units):
        for (p, _), g in zip(pred, gold):
            y_true.append(g)
            y_pred.append(p)
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
    return acc, f1_score(y_true, y_pred, average="weighted", zero_division=0)


def uncertain_share(predictions, threshold=0.9):
    """Share of symbols the model is unsure about (confidence below `threshold`)."""
    vals = [c for line in predictions for _, c in line]
    return sum(c < threshold for c in vals) / max(1, len(vals))


def save_model(path, model, hp, in_vocab, out_vocab, key, settings, metrics):
    torch.save({
        "format": FORMAT_NAME, "format_version": FORMAT_VERSION,
        # 16-bit weights halve the file size; predictions are unaffected.
        "model_state_dict": {k: (v.half() if v.is_floating_point() else v).cpu()
                             for k, v in model.state_dict().items()},
        "weights_dtype": "float16",
        "hyperparameters": hp, "input_vocab": in_vocab, "output_vocab": out_vocab,
        "key": key.to_json(), "settings": settings, "metrics": metrics,
    }, path)


def load_model(path, map_location=None):
    dev = map_location or device()
    try:
        ck = torch.load(path, map_location=dev, weights_only=False)
    except Exception as e:
        stop(f"The model file could not be read ({type(e).__name__}): {path}")
    if not isinstance(ck, dict) or ck.get("format") != FORMAT_NAME:
        stop(f"{path} is not a model made with this tool (version 1 or later). Older checkpoints "
             "from the paper experiments need retraining with the training notebook.")
    model = DecryptionLSTM(len(ck["input_vocab"]), len(ck["output_vocab"]),
                           pad_id=ck["input_vocab"][PAD], **ck["hyperparameters"])
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v)
                           for k, v in ck["model_state_dict"].items()})
    return model.to(dev), ck
