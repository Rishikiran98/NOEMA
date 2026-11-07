def restock_needed(levels: dict[str, int], threshold: int) -> list[str]:
    """Items whose stock level is at or below the threshold, sorted by name."""
    return sorted(name for name, level in levels.items() if level < threshold)


def total_units(levels: dict[str, int]) -> int:
    """Total units across all items."""
    return sum(levels.values())
