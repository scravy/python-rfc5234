from collections.abc import Iterator
from typing import final

from .expr import Expr
from frozenintset import FrozenIntSet


class ParseError(Exception):
    pass


class UnexpectedCharacter(ParseError):
    pass


class NoParse(ParseError):
    pass


class EvalError(NoParse):
    pass


class EmptyStack(EvalError):
    pass


class RequireFailed(EvalError):
    pass


@final
class NoAlternativeMatched(ExceptionGroup, NoParse):
    def __new__(cls, msg, exceptions):
        return super().__new__(cls, msg, exceptions)


@final
class EndOfFile(NoParse):
    def __init__(self, ix: int, expr: Expr, rs: FrozenIntSet):
        super().__init__(
            f"Expected one of {_render_rangeset(rs)}; reached end of file; pos={ix}; trying to apply {expr}"
        )


@final
class Expected(NoParse):
    def __init__(self, ix: int, expr: Expr, rs: FrozenIntSet, c: int):
        super().__init__(
            f"Expected one of {_render_rangeset(rs)}; looking at {_render_char(c)}; pos={ix}; trying to apply {expr}"
        )


def _render_char(c: int) -> str:
    if 0x20 < c < 0x7F:
        return f"'{chr(c)}'"
    return f"#x{c:02x}"


def _render_rs(rs: FrozenIntSet) -> Iterator[str]:
    for r in rs.ranges:
        lo = r.start
        hi = r.stop - 1
        if lo == hi:
            yield _render_char(lo)
        else:
            yield f"{_render_char(lo)}-{_render_char(hi)}"


def _render_rangeset(rs: FrozenIntSet) -> str:
    return ",".join(_render_rs(rs))
