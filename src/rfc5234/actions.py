from typing import final, NamedTuple

from .simple_expr import BoolExpr, IntExpr

type Action = Push | Pop | Set | Require


@final
class Push(NamedTuple):
    stack_name: str
    expr: IntExpr


@final
class Pop(NamedTuple):
    stack_name: str


@final
class Set(NamedTuple):
    counter_name: str
    expr: IntExpr


@final
class Require(NamedTuple):
    expr: BoolExpr
