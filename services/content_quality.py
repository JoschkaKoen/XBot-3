"""Free checks before media work; CEFR labels require a reviewed source entry."""
from pathlib import Path
import json
import os
import re
import unicodedata

QUALITY_INSTRUCTION = '''Correctness is a hard gate before humor: reject incorrect grammar,
missing German subordinate-clause commas, lowercase nominalized infinitives
(e.g. 'zum kochen'), unnatural wording, and mistranslated dictionary senses.
Teach whole expressions when needed (e.g. 'on hold', not 'hold'). Do not approve
literal Chinese such as '蛋糕享受' for a cake treat. The illustration must support
the taught situation; do not teach 'picnic' with a forest portrait showing no picnic.
Do not infer a word's CEFR level from the requested rotation slot.
'''


def normalize_word(word, language=''):
    word = unicodedata.normalize('NFKC', str(word or '')).casefold().strip()
    word = re.sub(r'^\s*(der|die|das|ein|eine)\s+', '', word) if language.lower() in ('german', 'de') else word
    word = re.sub(r'[^\w\s\-\'äöüß]', '', word)
    return ' '.join(word.split())


def lexicon(path=None):
    path = path or os.environ.get('BOT_CEFR_LEXICON', 'content/cefr_lexicon.json')
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def lemma(word, language):
    key = normalize_word(word, language)
    entry = lexicon().get(language.lower(), {}).get(key, {})
    return normalize_word(entry.get('lemma', key), language)


def used_words(records, language, days=None, now=None):
    # Revisions are opt-in. Default excludes the full history; an explicit review
    # interval never makes an invalid/missing timestamp qualify for repetition.
    from datetime import datetime, timezone
    now = now or datetime.now(timezone.utc)
    result = set()
    for row in records:
        if days:
            try:
                dt = datetime.fromisoformat(row['timestamp'])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if (now - dt).total_seconds() >= days * 86400:
                    continue
            except (KeyError, TypeError, ValueError):
                pass
        key = lemma(row.get('source_word', ''), language)
        if key:
            result.add(key)
    return result


def verified_cefr(word, language):
    entry = lexicon().get(language.lower(), {}).get(normalize_word(word, language), {})
    # Source provenance and human review are required. Model guesses are retained
    # as targets internally, never sent to learners as verified labels.
    if entry.get('reviewed') is True and str(entry.get('source', '')).startswith('https://'):
        level = entry.get('cefr', '')
        return level if level in ('A1','A2','B1','B2','C1','C2') else ''
    return ''


def issues(word, sentence, translation='', language='German'):
    result = []
    if not isinstance(sentence, str) or not sentence.strip():
        return ['missing sentence']
    normalized = normalize_word(sentence, language)
    target = normalize_word(word, language)
    entry = lexicon().get(language.lower(), {}).get(target, {})
    forms = {target, *(normalize_word(form, language) for form in entry.get("forms", []))}
    forms.discard("")
    if ' ' not in target:
        # Regular inflections only; irregular forms belong in the reviewed lexicon.
        if language.lower() in ('german', 'de') and target.endswith(('en', 'ern')):
            stem = target[:-2]
            forms.update(stem + suffix for suffix in ('e', 'st', 't', 'en', 'te', 'ten', 'test', 'tet'))
            if target.endswith('ern'):
                forms.update(target[:-1] + suffix for suffix in ('e', 'st', 't', 'n', 'te', 'ten'))
        elif language.lower() in ('english', 'en'):
            forms.update((target + 's', target + 'es', target + 'ed', target + 'ing'))
            if target.endswith('e'):
                forms.update((target + 'd', target[:-1] + 'ing'))
            if target.endswith('y'):
                forms.add(target[:-1] + 'ies')
    if not target or not any(re.search(r'(?<!\w)' + re.escape(form) + r'(?!\w)', normalized) for form in forms):
        result.append('taught word or phrase missing')
    if language.lower() in ('german', 'de'):
        if re.search(r'\b(?i:zum|beim)\s+[a-zäöüß][\wäöüß]*en\b', sentence):
            result.append('nominalized infinitive must be capitalized')
        if re.search(r'[\wäöüß] +(?:bevor|obwohl|weil|nachdem|damit)\b', sentence):
            result.append('subordinate clause requires a comma')
    if target == 'hold' and 'on hold' in normalized:
        result.append('teach the whole phrase on hold')
    if target == 'treat' and ('享受' in translation or '蛋糕享受' in translation):
        result.append('incorrect treat sense')
    return result


def reviewed_choice(raw, candidates):
    raw = raw.strip()
    if raw.startswith('```'):
        raw = '\n'.join(raw.splitlines()[1:-1])
    data = json.loads(raw)
    index = data.get('best_index')
    if data.get('approved') is not True or type(index) is not int or not 0 <= index < len(candidates):
        raise ValueError('No candidate passed the language quality review')
    return candidates[index]
