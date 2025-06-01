import itertools
import re
from collections.abc import Iterable, Iterator
from typing import NamedTuple, final
from frozenintset import FrozenIntSet

type Expr = Ref | Alt | Seq | Many | Char | Regex


@final
class Ref(NamedTuple):
    name: str

    def __str__(self) -> str:
        return self.name


@final
class Alt(NamedTuple):
    branches: tuple[Expr, ...]

    @staticmethod
    def of(*it: Iterable[Expr]) -> "Alt":
        return Alt(tuple(itertools.chain.from_iterable(it)))

    def __str__(self) -> str:
        return "/".join(f"({e})" if isinstance(e, (Seq, Alt)) else f"{e}" for e in self.branches)


@final
class Seq(NamedTuple):
    steps: tuple[Expr, ...]

    @staticmethod
    def of(*it: Iterable[Expr]) -> "Seq":
        return Seq(tuple(itertools.chain.from_iterable(it)))

    def __str__(self) -> str:
        return "/".join(f"({e})" if isinstance(e, (Seq, Alt)) else f"{e}" for e in self.steps)


@final
class Many(NamedTuple):
    max: int | None  # None = infinite
    expr: Expr

    def __str__(self) -> str:
        expr = f"({self.expr})" if isinstance(self.expr, (Seq, Alt)) else f"{self.expr}"
        return f"*{self.max or ''}{expr}"


@final
class Char(NamedTuple):
    allowed: FrozenIntSet

    def __str__(self) -> str:
        return "/".join(
            (f"%x{rng.start:02x}" if rng.start == rng.stop - 1 else f"%x{rng.start:02x}-{rng.stop - 1:02x}")
            for rng in self.allowed.ranges
        )


@final
class Regex(NamedTuple):
    pattern: re.Pattern[str]

    @staticmethod
    def of(pattern: str) -> "Regex":
        return Regex(re.compile(pattern))


def refs(expr: Expr) -> Iterator[str]:
    match expr:
        case Many(expr=e):
            yield from refs(e)
        case Alt(es) | Seq(es):
            for e in es:
                yield from refs(e)
        case Ref(rule):
            yield rule
