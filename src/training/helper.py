




def build_vocab(patients: list[dict], pad_idx: int) -> dict[str, int]:
    """Map every unique ICD code and med name to a positive integer index.
    Index 0 is reserved for [PAD].
    """
    print(f"[build_vocab] building vocab ([PAD]: {pad_idx})...")
    tokens: set[str] = set()
    for p in patients:
        for enc in p.get("encounters", []):
            tokens.update(enc.get("icd_codes", []))
            tokens.update(enc.get("meds", []))
    vocab: dict[str, int] = {"[PAD]": pad_idx}
    for i, tok in enumerate(sorted(tokens), start=1):
        vocab[tok] = i
        
    print(f"   len: {len(vocab)}")
    return vocab


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