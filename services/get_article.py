import logging
from german_nouns.lookup import Nouns

logger = logging.getLogger("german_bot.article")

_nouns = Nouns()

_GENUS_MAP = {"m": "der", "f": "die", "n": "das"}


def get_article(word: str) -> str:
    """
    Look up the grammatical article for a German noun.

    Returns "der", "die", "das", or "no known noun".
    """
    result = _nouns[word]
    if not result:
        logger.debug("No article found for '%s'", word)
        return "no known noun"

    genus = result[0].get("genus", "")
    article = _GENUS_MAP.get(genus, "no known noun")
    logger.debug("Article for '%s': %s (genus=%s)", word, article, genus)
    return article
