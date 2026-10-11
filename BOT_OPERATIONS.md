Bot reliability and quality changes
==================================

Install only through an explicitly authorized deployment. Preserve Ubuntu's
settings.env, credentials, data, media, venv, and existing uncommitted edits.
The tracked deploy/xbot3.service includes bounded console output. After code
installation, install the unit and reload systemd, then restart BETWEEN cycles.
Do not stop a live media render/publication to roll out these changes.

Normal operation
- data/health.json records stage, heartbeat, next slot, last successful X write,
  failures and per-destination health. Waiting for the next slot is healthy.
- An exclusive kernel lock survives exec-based updates. Proxy readiness and
  a 2 GiB free-disk floor precede generation (BOT_MIN_FREE_BYTES overrides it).
- A durable per-account delivery journal prevents duplicate writes. Typed HTTP
  402 is attempted once, then held for an hour. Transport/5xx response loss and
  an interrupted write are held indefinitely for operator reconciliation.
- Do not blindly clear a posting/uncertain journal. Inspect X for the saved
  cycle/text and media. If confirmed published, record status=ok, the real ID,
  URL and last_success_at. If positively confirmed absent, set status=rejected.
  Make a backup before editing. Successful primary journals recover missing
  history after a post/save crash without contacting X.
- Secondary X failure does not block its compliance-approved website mirror.

Quality and cadence
- Default deduplication covers all retained normalized word history. Reviewed
  lexicon lemmas/forms handle lexical variants. VOCAB_REVIEW_DAYS in a secondary
  account opts into deliberate revision; missing timestamps stay excluded.
- Existing selection/translation calls now require a positive language-quality
  verdict before expensive media. Free rules reject the audited grammar/sense
  errors. They cannot prove every sentence correct.
- Secondary examples receive a separate simplicity review before selection:
  common beginner words around the taught word, simple grammar, and a concrete
  context that makes its meaning clear. The same review requires an easy joke and
  a specific visible image detail; generic captions and invented visible actions
  are rejected. Simplify the joke rather than dropping humor. Up to three
  lesson attempts are allowed; exhausted or malformed reviews skip this target.
- Chinese posts put tone-marked pinyin beside the Chinese word meaning and below
  the Chinese sentence. Pronunciation is generated and checked by the model,
  with local format checks; it is not a dictionary-based pronunciation guarantee.
  Shorter translations replace Chinese and pinyin together. Optional decoration
  goes first; oversized lessons get shorter, reviewed examples, never text slices.
  History and website metadata preserve audience_word_pinyin and
  audience_sentence_pinyin as separate fields. Existing posts are not backfilled.
- CEFR rotation is a generation target, not learner-facing evidence. The initial
  content/cefr_lexicon.json is intentionally empty: no unverified CEFR label is
  displayed. BOT_CEFR_LEXICON can select a reviewed dictionary with language keys
  german/english and normalized-word entries containing cefr, source HTTPS URL,
  reviewed=true, optional lemma and forms. Populate only from verified sources.
- Images are ranked against the visible taught situation. Secondary review
  checks compatibility with the shared scene; this is not a pixel-level proof.
  Review naturally generated stills before asserting the picnic problem solved.

Measurement and storage
Run python scripts/bot_report.py --root . (offline). Reports sample sizes and
observed ages within 24–36h and 168–180h windows, per account and destination.
Missing observations are unavailable, not zero. At least 30 comparable 24h
observations are required before the automatic strategist can change strategy.
Continue the current cadence; compare one content variable at a time. No new
paid experiments or additional image candidates are enabled by this change.

Console logs keep at most four 10 MiB files. The inventory reports generated
media sizes and free disk. Preserve published files and all data. A future
archive pass should first produce a manifest of unreferenced generated assets,
verify it against histories and website references, then move those assets to
an archive with a restore path; no generated-media deletion is automatic.

Verification
Offline unittest checks cover billing, response-loss holds, successful-write
idempotence, confirmed-write history recovery, subprocess singleton exclusion,
schedules, cited language errors, label provenance, fixed-age request budgets
and log rotation. No live AI or publication smoke tests are required.
