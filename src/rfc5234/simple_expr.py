from collections.abc import Iterator
from enum import Enum
from typing import final, NamedTuple

from .context import Context
from .exc import EmptyStack


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


def eval_bool_expr(ctx: Context, pos: int, expr: BoolExpr) -> bool:
    match expr:
        case And(left, right):
            return eval_bool_expr(ctx, pos, left) and eval_bool_expr(ctx, pos, right)
        case Or(left, right):
            return eval_bool_expr(ctx, pos, left) or eval_bool_expr(ctx, pos, right)
        case Not(expr):
            return not eval_bool_expr(ctx, pos, expr)
        case Comparison(op, left, right):
            left_result = eval_int_expr(ctx, pos, left)
            right_result = eval_int_expr(ctx, pos, right)
            match op:
                case ComparisonOp.EQ:
                    return left_result == right_result
                case ComparisonOp.NEQ:
                    return left_result != right_result
                case ComparisonOp.LT:
                    return left_result < right_result
                case ComparisonOp.LTE:
                    return left_result <= right_result
                case ComparisonOp.GT:
                    return left_result > right_result
                case ComparisonOp.GTE:
                    return left_result >= right_result
        case EmptyFn(stack_name):
            return ctx.top(stack_name) is None
    raise RuntimeError("unreachable code")


def eval_int_expr(ctx: Context, pos: int, expr: IntExpr) -> int:
    match expr:
        case IntApp1(fn, arg):
            val = eval_int_expr(ctx, pos, arg)
            match fn:
                case IntFn1.ABS:
                    return abs(val)
            raise RuntimeError(f"unknown unary function {fn}")
        case IntApp2(fn, arg1, arg2):
            arg1_value = eval_int_expr(ctx, pos, arg1)
            arg2_value = eval_int_expr(ctx, pos, arg2)
            match fn:
                case IntFn2.MIN:
                    return min(arg1_value, arg2_value)
                case IntFn2.MAX:
                    return max(arg1_value, arg2_value)
                case IntFn2.ADD:
                    return arg1_value + arg2_value
                case IntFn2.SUB:
                    return arg1_value - arg2_value
                case IntFn2.MUL:
                    return arg1_value * arg2_value
            raise RuntimeError(f"unknown binary function {fn}")
        case TopFn(stack_name):
            top = ctx.top(stack_name)
            if top is not None:
                return top
            raise EmptyStack(stack_name)
        case CounterRef(counter_name):
            return ctx.get(counter_name)
        case SpecialRef("pos"):
            return pos
        case SpecialRef():
            raise RuntimeError("unknown special reference")
        case IntLit(value):
            return value
    raise RuntimeError("unreachable code")


def stack_refs(expr: IntExpr | BoolExpr) -> frozenset[str]:
    return frozenset(_stack_refs(expr))


def _stack_refs(expr: IntExpr | BoolExpr) -> Iterator[str]:
    match expr:
        case IntApp1(arg1=arg):
            yield from _stack_refs(arg)
        case IntApp2(arg1=arg1, arg2=arg2):
            yield from _stack_refs(arg1)
            yield from _stack_refs(arg2)
        case TopFn(stack_name):
            yield stack_name
        case And(left=left, right=right):
            yield from _stack_refs(left)
            yield from _stack_refs(right)
        case Or(left=left, right=right):
            yield from _stack_refs(left)
            yield from _stack_refs(right)
        case Not(expr=expr):
            yield from _stack_refs(expr)
        case Comparison(left=left, right=right):
            yield from _stack_refs(left)
            yield from _stack_refs(right)
        case EmptyFn(stack_name):
            yield stack_name


def counter_refs(expr: IntExpr | BoolExpr) -> frozenset[str]:
    return frozenset(_counter_refs(expr))


def _counter_refs(expr: IntExpr | BoolExpr) -> Iterator[str]:
    match expr:
        case IntApp1(arg1=arg):
            yield from _counter_refs(arg)
        case IntApp2(arg1=arg1, arg2=arg2):
            yield from _counter_refs(arg1)
            yield from _counter_refs(arg2)
        case CounterRef(counter_name):
            yield counter_name
        case And(left=left, right=right):
            yield from _counter_refs(left)
            yield from _counter_refs(right)
        case Or(left=left, right=right):
            yield from _counter_refs(left)
            yield from _counter_refs(right)
        case Not(expr=expr):
            yield from _counter_refs(expr)
        case Comparison(left=left, right=right):
            yield from _counter_refs(left)
            yield from _counter_refs(right)
