import logging
from german_nouns.lookup import Nouns

logger = logging.getLogger("german_bot.article")

_nouns = Nouns()

_GENUS_MAP = {"m": "der", "f": "die", "n": "das"}

# Common German plural suffixes — when a word ends in one of these AND the
# noun dictionary returns a masculine/neuter article (meaning the singular was
# looked up), the word is almost certainly a plural and the article is "die".
_PLURAL_SUFFIXES = ("en", "nen", "innen", "er", "e", "s")


def _looks_like_plural(word: str, singular_article: str) -> bool:
    """
    Heuristic: if the lookup returned a masculine/neuter article but the word
    ends in a typical plural suffix, it's likely a plural form → article = "die".
    """
    if singular_article not in ("der", "das"):
        return False  # already "die" or unknown — no mismatch to resolve
    w = word.lower()
    return any(w.endswith(s) for s in _PLURAL_SUFFIXES)


def get_article(word: str) -> str:
    """
    Look up the grammatical article for a German noun.

    Returns "der", "die", "das", or "no known noun".
    Plural forms (which always take "die") are detected via a suffix heuristic.
    """
    result = _nouns[word]
    if not result:
        logger.debug("No article found for '%s'", word)
        return "no known noun"

    genus = result[0].get("genus", "")
    article = _GENUS_MAP.get(genus, "no known noun")

    if _looks_like_plural(word, article):
        logger.debug("'%s' looks like a plural — overriding %s → die", word, article)
        article = "die"

    logger.debug("Article for '%s': %s (genus=%s)", word, article, genus)
    return article
