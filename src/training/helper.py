import torch
from torch.amp import GradScaler # type: ignore


def build_vocab(patients: list[dict], pad_idx: int) -> dict[str, int]:
    """Map every unique ICD code and med name to a positive integer index.
    Index 0 is reserved for [PAD].
    """
    tokens: set[str] = set()
    for p in patients:
        for enc in p.get("encounters", []):
            tokens.update(enc.get("icd_codes", []))
            tokens.update(enc.get("meds", []))
    vocab: dict[str, int] = {"[PAD]": pad_idx}
    for i, tok in enumerate(sorted(tokens), start=1):
        vocab[tok] = i
        
    print(f"[build_vocab] size: {len(vocab)}")
    return vocab



def init_model(device, params):
    from src.models.sequential_jepa import JEPA
    model = JEPA(**params).to(device)
    return model


def init_optimizers(
    model,
    ipe: int,
    num_epochs: int,
    opt_params={},
    use_bfloat16=False
):
    base_lr = opt_params.get("base_lr", 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    
    sched_type = opt_params.get("schedule", "")
    if sched_type == "warmup_cosine_annealing":
        from src.training.schedulers import WarmupCosineAnnealing
        scheduler = WarmupCosineAnnealing(
            optimizer,
            warmup_steps=ipe*opt_params.get("warmup_epochs", 0),
            total_steps=ipe*num_epochs,
            min_lr=base_lr)
    elif sched_type == "linear_decay":
        total_steps = ipe*num_epochs
        min_lr_ratio = opt_params.get("min_lr_ratio", 0.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: max(min_lr_ratio, 
                                       1.0 - (1.0 - min_lr_ratio) * step / total_steps))
    elif sched_type == "linear":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: base_lr)
    else:
        raise ValueError(f"[init_optimizers] Unknown scheduler '{sched_type}'. Expected 'warmup_cosine_annealing', 'linear_decay', or 'linear'")
    
    scaler = GradScaler('cuda') if use_bfloat16 else None
    return optimizer, scheduler, scaler


class CSVLogger(object):
    def __init__(self, fname, *argv):
        self.fname = fname
        self.types = []
        # -- print headers
        with open(self.fname, '+a') as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=',', file=f)
                else:
                    print(v[1], end='\n', file=f)

    def log(self, *argv):
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)



class ValMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count