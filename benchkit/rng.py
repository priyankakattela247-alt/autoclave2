"""Integer-only PRNG and seed derivation. No floats, no `random`."""

import hashlib
import json

_MASK = (1 << 64) - 1


def derive_seed(*parts) -> int:
    text = "\x1f".join(str(p) for p in parts)
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")


def digest_ints(ints, profile=None) -> str:
    """16-hex fingerprint of an integer sequence, optionally folding in a profile."""
    h = hashlib.blake2b(digest_size=8)
    if profile is not None:
        h.update(json.dumps(dict(profile), sort_keys=True).encode())
        h.update(b"\x00")
    h.update(",".join(str(int(i)) for i in ints).encode())
    return h.hexdigest()


class Rng:
    """SplitMix64."""

    def __init__(self, seed: int):
        self.state = seed & _MASK

    def next64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        """Uniform in [0, n) by rejection sampling (no modulo bias)."""
        if n <= 0:
            raise ValueError("below(n) needs n > 0")
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            r = self.next64()
            if r < limit:
                return r % n

    def between(self, a: int, b: int) -> int:
        """Uniform in [a, b] inclusive."""
        return a + self.below(b - a + 1)

    def choice(self, seq):
        return seq[self.below(len(seq))]

    def shuffle(self, items: list) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]

    def chance(self, num: int, den: int) -> bool:
        return self.below(den) < num
