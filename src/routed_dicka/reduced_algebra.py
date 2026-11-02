"""Real-part word algebra for protocol-specific reduced BFF models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Iterable


Word = tuple[int, ...]


@dataclass(frozen=True)
class Symbol:
    name: str
    party: str
    setting: str
    outcome: int
    kind: str = "projector"
    adj: int = -1


class Algebra:
    """Projection quotient with explicitly declared inter-party commutation."""

    def __init__(
        self,
        symbols: Iterable[Symbol],
        commuting_parties: Iterable[tuple[str, str]],
    ) -> None:
        self.symbols = tuple(symbols)
        self.cp = {frozenset(pair) for pair in commuting_parties}
        self.n = len(self.symbols)
        self.star = tuple(
            index if symbol.adj < 0 else symbol.adj
            for index, symbol in enumerate(self.symbols)
        )
        self.is_projector = tuple(
            symbol.kind == "projector" for symbol in self.symbols
        )

    def commute(self, first: int, second: int) -> bool:
        parties = frozenset(
            (self.symbols[first].party, self.symbols[second].party)
        )
        return first == second or parties in self.cp

    @lru_cache(maxsize=500_000)
    def word(self, value: Word | None) -> Word | None:
        if value is None:
            return None
        letters = list(value)
        while True:
            changed = False
            for left_index, left_symbol_index in enumerate(letters):
                if not self.is_projector[left_symbol_index]:
                    continue
                left = self.symbols[left_symbol_index]
                for right_index in range(left_index + 1, len(letters)):
                    right_symbol_index = letters[right_index]
                    right = self.symbols[right_symbol_index]
                    if (
                        self.is_projector[right_symbol_index]
                        and (left.party, left.setting) == (right.party, right.setting)
                    ):
                        if left.outcome != right.outcome:
                            return None
                        del letters[right_index]
                        changed = True
                        break
                    if not self.commute(left_symbol_index, right_symbol_index):
                        break
                if changed:
                    break
            if not changed:
                break

        remaining = list(range(len(letters)))
        ordered: list[int] = []
        while remaining:
            ready = [
                candidate
                for candidate in remaining
                if all(
                    predecessor > candidate
                    or self.commute(letters[predecessor], letters[candidate])
                    for predecessor in remaining
                    if predecessor != candidate
                )
            ]
            chosen = min(ready, key=lambda index: (letters[index], index))
            ordered.append(letters[chosen])
            remaining.remove(chosen)
        return tuple(ordered)

    def adjoint(self, word: Word) -> Word:
        return self.word(tuple(self.star[index] for index in reversed(word))) or ()

    def moment(self, word: Word) -> Word | None:
        reduced = self.word(word)
        if reduced is None:
            return None
        return min(reduced, self.adjoint(reduced))

    def label(self, word: Word) -> str:
        return "I" if not word else " ".join(self.symbols[index].name for index in word)

    def z_degree(self, word: Word) -> int:
        return sum(not self.is_projector[index] for index in word)

    def basis(self, level: int = 2, enrich: Iterable[Word] = ()) -> list[Word]:
        words: set[Word | None] = {()}
        for degree in range(1, level + 1):
            words.update(self.word(word) for word in product(range(self.n), repeat=degree))
        words.update(self.word(word) for word in enrich)
        words.discard(None)
        return sorted(words, key=lambda word: (len(word), word))  # type: ignore[arg-type]


def add_term(polynomial: dict[Word, int | float], word: Word | None, value) -> None:
    if word is not None and value:
        polynomial[word] = polynomial.get(word, 0) + value
    if word in polynomial and polynomial[word] == 0:
        del polynomial[word]


def bff_polynomials(
    algebra: Algebra,
    alice_key_projector: int,
    eve_operators: Iterable[int],
) -> tuple[dict[Word, int], dict[Word, int]]:
    """Return exact coefficients ``p0 + t*p1`` for the BFF objective."""

    p0: dict[Word, int] = {}
    p1: dict[Word, int] = {}
    for outcome, z in enumerate(eve_operators):
        z_dagger = algebra.star[z]
        key_terms = (
            [((alice_key_projector,), 1)]
            if outcome == 0
            else [((), 1), ((alice_key_projector,), -1)]
        )
        for key_word, sign in key_terms:
            for suffix in ((z,), (z_dagger,), (z_dagger, z)):
                add_term(p0, algebra.moment(key_word + suffix), sign)
            add_term(p1, algebra.moment(key_word + (z_dagger, z)), -sign)
        add_term(p1, algebra.moment((z, z_dagger)), 1)
    return p0, p1
