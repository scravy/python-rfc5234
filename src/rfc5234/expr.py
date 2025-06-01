import itertools
import re
from collections.abc import Iterable, Iterator, Callable, Set
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


def all_refs(resolve: Callable[[str], Expr], rulename: str) -> frozenset[str]:
    def _all_refs(name: str, visited: set[str]) -> Iterator[str]:
        visited.add(name)
        expr = resolve(name)
        for ref in refs(expr):
            yield ref
            if ref not in visited:
                yield from _all_refs(ref, visited)

    return frozenset(_all_refs(rulename, set()))


def inline(expr: Expr, resolve: Callable[[str], Expr], rule: str) -> Expr:
    match expr:
        case Ref(rn) if rn == rule:
            return resolve(rn)
        case Alt(es):
            return Alt.of(inline(e, resolve, rule) for e in es)
        case Seq(es):
            return Seq.of(inline(e, resolve, rule) for e in es)
        case Many(max=n, expr=e):
            return Many(max=n, expr=inline(e, resolve, rule))
    return expr


def can_be_empty(expr: Expr, nullable: Set[str]) -> bool:
    match expr:
        case Many():
            return True
        case Alt(branches):
            return any(can_be_empty(b, nullable) for b in branches)
        case Seq(steps):
            return all(can_be_empty(s, nullable) for s in steps)
        case Ref(name):
            return name in nullable
        case Char():
            return False
        case Regex():
            raise NotImplementedError("first_set not implemented for Regex")
    raise RuntimeError("unreachable code")


def first_set(expr: Expr, resolve: Callable[[str], Expr], nullable: Set[str]) -> FrozenIntSet:
    known: dict[str, FrozenIntSet] = dict()

    def _first_set(e: Expr):
        match e:
            case Ref(rn):
                if rn not in known:
                    known[rn] = FrozenIntSet()
                    known[rn] = _first_set(resolve(rn))
                return known[rn]
            case Alt(es):
                return FrozenIntSet.union_all(_first_set(e) for e in es)
            case Seq(es):
                rs = []
                for e in es:
                    rs.append(_first_set(e))
                    if not can_be_empty(e, nullable):
                        break
                return FrozenIntSet.union_all(rs)
            case Many(expr=e):
                return _first_set(e)
            case Char(cs):
                return cs
            case Regex():
                raise NotImplementedError("first_set not implemented for Regex")
        raise RuntimeError("unreachable code")

    return _first_set(expr)
