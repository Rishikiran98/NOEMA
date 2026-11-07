import re


def slugify(title: str, max_length: int = 40) -> str:
    """Lowercase, words joined by '-', non-alphanumerics dropped, at most max_length characters,
    never ending in '-'."""
    words = re.findall(r"[a-z0-9]+", title.lower())
    slug = "-".join(words)
    if len(slug) > max_length:
        slug = slug[:max_length]
    return slug
