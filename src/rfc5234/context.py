from collections import ChainMap
from typing import NamedTuple, final, Self

from .exc import EmptyStack

type Stack = _Stack | None


class _Stack(NamedTuple):
    val: int
    tail: Stack


@final
class Context:
    __slots__ = ("_stacks", "_counters")

    def __init__(
        self,
        stacks: ChainMap[str, Stack],
        counters: ChainMap[str, int],
    ):
        self._stacks: ChainMap[str, Stack] = stacks
        self._counters: ChainMap[str, int] = counters

    def push(self, stack_name: str, value: int) -> None:
        stack = self._stacks[stack_name]
        self._stacks[stack_name] = _Stack(value, stack)

    def pop(self, stack_name: str) -> None:
        stack = self._stacks[stack_name]
        if stack is None:
            raise EmptyStack(stack_name)
        self._stacks[stack_name] = stack.tail

    def top(self, stack_name) -> int | None:
        stack = self._stacks[stack_name]
        if stack is None:
            return None
        return stack.val

    def set(self, counter_name: str, new_value: int) -> None:
        self._counters[counter_name] = new_value

    def get(self, counter_name: str) -> int:
        return self._counters[counter_name]

    def fork(self) -> Self:
        return Context(
            stacks=ChainMap({}, *self._stacks.maps),
            counters=ChainMap({}, *self._counters.maps),
        )

    def join(self, ctx: Self) -> None:
        for name, state in ctx._stacks.items():
            self._stacks[name] = state
        for name, value in ctx._counters.items():
            self._counters[name] = value
