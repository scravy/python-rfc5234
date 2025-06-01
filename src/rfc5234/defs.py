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

    def __call__(
        self,
        *,
        events: None | ParseEventKind = None,
        events_filter: None | re.Pattern[str] = None,
        strategy: None | ParsingStrategy = None,
        use_regex: bool | None = None,
        use_cache: bool | None = None,
        inline_strategy: InlineStrategy | None = None,
    ):
        return Opts(
            events=self.events if events is None else events,
            events_filter=self.events_filter if events_filter is None else events_filter,
            strategy=self.strategy if strategy is None else strategy,
            use_regex=self.use_regex if use_regex is None else use_regex,
            use_cache=self.use_cache if use_cache is None else use_cache,
            inline_strategy=self.inline_strategy if inline_strategy is None else inline_strategy,
        )
