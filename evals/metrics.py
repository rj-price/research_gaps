"""Scoring primitives shared by the suites.

Deliberately dependency-free and exercised by `evals/test_metrics.py`: a harness that
miscounts is worse than no harness at all.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass
class BinaryScore:
    """Confusion matrix for a yes/no decision, with the usual derived rates."""
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, predicted: bool, actual: bool) -> None:
        if predicted and actual:
            self.tp += 1
        elif predicted and not actual:
            self.fp += 1
        elif not predicted and actual:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return _ratio(self.tp + self.tn, self.total)

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return _ratio(2 * p * r, p + r) if (p + r) else 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
        }


@dataclass
class SetScore:
    """Micro-averaged precision and recall over set-valued predictions.

    `ignored` holds items excluded from scoring entirely — the defensible-but-not-required
    gap matches — so a sensible extra link is not punished as a false positive.
    """
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ignored: int = 0

    def add(self, predicted: Iterable[str], expected: Iterable[str], allowed: Iterable[str] = ()) -> None:
        predicted, expected, allowed = set(predicted), set(expected), set(allowed)
        scored = predicted - (allowed - expected)
        self.ignored += len(predicted & (allowed - expected))
        self.tp += len(scored & expected)
        self.fp += len(scored - expected)
        self.fn += len(expected - scored)

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return _ratio(2 * p * r, p + r) if (p + r) else 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "ignored": self.ignored,
        }


@dataclass
class Mean:
    """Running mean, reported to two decimals alongside its sample count."""
    values: List[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.values.append(float(value))

    @property
    def mean(self) -> float:
        return _ratio(sum(self.values), len(self.values))

    def as_dict(self, name: str) -> Dict[str, float]:
        return {name: round(self.mean, 4), f"{name}_n": len(self.values)}
