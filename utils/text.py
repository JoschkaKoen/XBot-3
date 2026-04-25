"""
utils/text — small, pure string helpers used by tweet post-processing.

These are deliberately not in nodes/generate_content.py: they are pure
functions with no project state, easy to unit-test, and may be reused by
verification or analytics code in the future.
"""


def truncate_emoji_pairs(tweet: str) -> str:
    """
    Replace doubled trailing emoji pairs (🚗🚗 → 🚗) on each tweet line.

    The tweet scaffold ends every line with two spaces and an emoji pair; the
    AI sometimes duplicates that pair for emphasis (e.g. "🚗🚗"), pushing the
    tweet past MAX_TWEET_LENGTH. This collapses the doubled half back to a
    single occurrence — meaning is preserved.

    Args:
        tweet: full tweet text, possibly with doubled trailing emoji pairs.

    Returns:
        The tweet with doubled trailing emoji pairs collapsed.
    """
    lines = tweet.split("\n")
    result = []
    for line in lines:
        if "  " in line:
            idx = line.rfind("  ")
            prefix = line[: idx + 2]
            suffix = line[idx + 2 :].strip()
            n = len(suffix)
            if n % 2 == 0 and n > 0 and suffix[: n // 2] == suffix[n // 2 :]:
                line = prefix + suffix[: n // 2]
        result.append(line)
    return "\n".join(result)
