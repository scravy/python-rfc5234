import re
from enum import auto, IntFlag, IntEnum
from typing import NamedTuple, final


@final
class ParseEventKind(IntFlag):
    START = auto()
    VALUE = auto()
    END = auto()


@final
class ParseEvent(NamedTuple):
    kind: ParseEventKind
    rule: str
    pos: int
    value: str = ""


@final
class ParsingStrategy(IntEnum):
    FIRST_MATCH = auto()
    LONGEST_MATCH = auto()


@final
class InlineStrategy(IntEnum):
    NO_INLINE = auto()
    EXPLICIT = auto()
    AGGRESSIVE = auto()


@final
class Opts(NamedTuple):
    events: None | ParseEventKind = None
    events_filter: None | re.Pattern[str] = None
    strategy: None | ParsingStrategy = None
    use_regex: bool | None = None
    use_cache: bool | None = None
    inline_strategy: InlineStrategy | None = None
