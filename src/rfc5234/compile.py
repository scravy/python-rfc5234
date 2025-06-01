import re
from collections.abc import Callable

from typing import NamedTuple

from frozenintset import FrozenIntSet

from .expr import (
    Alt,
    Char,
    Expr,
    Many,
    Regex,
    Seq,
)


class CompilationContext(NamedTuple):
    dedup: dict[Expr, Expr]
    use_regex: bool


def compile_rules(rules: dict[str, Expr], *, use_regex: bool) -> dict[str, Expr]:
    ctx = CompilationContext(dedup=dict(), use_regex=use_regex)
    return {name: compile_expr(ctx, expr) for name, expr in rules.items()}


def flatten_and_compile(
    extract: Callable[[Expr], tuple[Expr, ...] | None],
    ctx: CompilationContext,
    es: tuple[Expr, ...],
) -> tuple[Expr, ...]:
    esc: list[Expr] = list()
    for e in es:
        ec = compile_expr(ctx, e)
        if (ec_es := extract(ec)) is not None:
            esc.extend(ec_es)
        else:
            esc.append(ec)
    return tuple(esc)


def compile_expr(ctx: CompilationContext, expr: Expr) -> Expr:
    match expr:
        case Alt(es):
            ces = flatten_and_compile(
                lambda ex: ex.branches if isinstance(ex, Alt) else None,
                ctx,
                es,
            )
            if all(isinstance(ce, Char) for ce in ces):
                expr = Char(FrozenIntSet.union_all(ce.allowed for ce in ces))  # type: ignore[union-attr]
            else:
                expr = Alt(ces)
        case Seq(es):
            expr = Seq(
                flatten_and_compile(
                    lambda ex: ex.steps if isinstance(ex, Seq) else None,
                    ctx,
                    es,
                )
            )
        case Many(max=max_, expr=e):
            expr = Many(max=max_, expr=compile_expr(ctx, e))
    if ctx.use_regex:
        expr = compile_to_regex(expr)
    if expr not in ctx.dedup:
        ctx.dedup[expr] = expr
    return ctx.dedup[expr]


def regex_escape(c: int) -> str:
    match c:
        case 0x20 | 0x2A | 0x2B | 0x2D | 0x3F | 0x5B | 0x5D:
            return rf"\x{c:02x}"
    return re.escape(chr(c))


# noinspection RegExpUnnecessaryNonCapturingGroup
def compile_to_regex(expr: Expr) -> Expr:
    match expr:
        case Alt(es) if all(isinstance(e, Regex) for e in es):
            expr = Regex.of("(" + "|".join(e.pattern.pattern for e in es) + ")")  # type: ignore[union-attr]
        case Seq(es) if all(isinstance(e, Regex) for e in es):
            expr = Regex.of("(" + "".join(e.pattern.pattern for e in es) + ")")  # type: ignore[union-attr]
        case Many(max=1, expr=Regex(pat)):
            expr = Regex.of(f"({pat.pattern})?")
        case Many(max=None, expr=Regex(pat)):
            expr = Regex.of(f"({pat.pattern})*")
        case Many(max=max_, expr=Regex(pat)):
            rep = "".join(["{0,", "" if max_ is None else str(max_), "}"])
            expr = Regex.of(f"({pat.pattern}){rep}")
        case Char(rs):
            rx = "".join(
                (
                    regex_escape(rng.start)
                    if rng.start == rng.stop - 1
                    else f"{regex_escape(rng.start)}-{regex_escape(rng.stop - 1)}"
                )
                for rng in rs.ranges
            )
            try:
                expr = Regex.of(rx if len(rx) == 1 else f"[{rx}]")
            except re.error:
                pass
    return expr
