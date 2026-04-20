from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class TMEntry:
    source_text: str
    target_text: str
    hit_count: int = 0


class TranslationMemory:
    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self._store: OrderedDict[tuple[str, str, str], TMEntry] = OrderedDict()

    def upsert(self, source_lang: str, target_lang: str, normalized_source: str, target_text: str) -> None:
        key = (source_lang, target_lang, normalized_source)
        existing = self._store.get(key)
        if existing is None:
            self._store[key] = TMEntry(source_text=normalized_source, target_text=target_text, hit_count=0)
        else:
            existing.target_text = target_text
        self._store.move_to_end(key)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)

    def get_exact(self, source_lang: str, target_lang: str, normalized_source: str) -> str | None:
        key = (source_lang, target_lang, normalized_source)
        entry = self._store.get(key)
        if entry is None:
            return None
        entry.hit_count += 1
        self._store.move_to_end(key)
        return entry.target_text

    def iter_pair_entries(self, source_lang: str, target_lang: str) -> list[TMEntry]:
        return [
            entry
            for (src, tgt, _), entry in self._store.items()
            if src == source_lang and tgt == target_lang
        ]

    def clear(self) -> None:
        self._store.clear()
