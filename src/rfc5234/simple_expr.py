from enum import Enum
from typing import final, NamedTuple

type IntExpr = IntApp1 | IntApp2 | TopFn | CounterRef | SpecialRef | IntLit


@final
class IntFn1(Enum):
    ABS = "abs"


@final
class IntApp1(NamedTuple):
    fn: IntFn1
    arg1: IntExpr


@final
class IntFn2(Enum):
    MIN = "min"
    MAX = "max"
    ADD = "+"
    SUB = "-"
    MUL = "*"


@final
class IntApp2(NamedTuple):
    fn: IntFn2
    arg1: IntExpr
    arg2: IntExpr


@final
class TopFn(NamedTuple):
    stack_ref: str


@final
class CounterRef(NamedTuple):
    counter_ref: str


@final
class SpecialRef(NamedTuple):
    special_ref: str


@final
class IntLit(NamedTuple):
    value: int


type BoolExpr = And | Or | Not | Comparison | EmptyFn


@final
class And(NamedTuple):
    left: BoolExpr
    right: BoolExpr


@final
class Or(NamedTuple):
    left: BoolExpr
    right: BoolExpr


@final
class Not(NamedTuple):
    expr: BoolExpr


@final
class ComparisonOp(Enum):
    EQ = "="
    NEQ = "!="
    LTE = "<="
    LT = "<"
    GTE = ">="
    GT = ">"


@final
class Comparison(NamedTuple):
    op: ComparisonOp
    left: IntExpr
    right: IntExpr


@final
class EmptyFn(NamedTuple):
    stack_ref: str
