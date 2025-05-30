import sys
from typing import NamedTuple, final, TextIO


type AST = ASTNode | ASTLeaf


@final
class ASTNode(NamedTuple):
    kind: str
    children: tuple["AST", ...]

    def to_json(self):
        return {self.kind: [child.to_json() for child in self.children]}

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            for child in self.children:
                if child.kind == item:
                    return True
        return False

    def render(self, fp: TextIO = sys.stdout, indent: int = 0) -> None:
        # noinspection PyTypeChecker
        print(indent * "  ", f"{self.kind}:", sep="", file=fp)
        for child in self.children:
            child.render(fp, indent + 1)


@final
class ASTLeaf(NamedTuple):
    kind: str
    value: str

    def to_json(self):
        return {self.kind: self.value}

    def render(self, fp: TextIO = sys.stdout, indent: int = 0) -> None:
        # noinspection PyTypeChecker
        print(indent * "  ", f"{self.kind}: {self.value}", sep="", file=fp)


@final
class ASTBuilder(NamedTuple):
    kind: str
    start: int
    children: list[AST]

    def build(self, source: str, pos: int) -> AST:
        if self.children:
            return ASTNode(self.kind, tuple(self.children))
        return ASTLeaf(self.kind, source[self.start : pos])
