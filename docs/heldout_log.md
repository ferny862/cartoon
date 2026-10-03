# Held-out period log

The held-out period starts **2021-10-01** (approved 2026-10-03) and runs to the
latest available data. Its results must not be viewed until the final
configuration is approved.

The data layer enforces this: analysis data on or after the held-out start is
withheld, and requesting it raises `HeldOutLockedError`, until
`dates.heldout_unlocked` in `config/settings.yaml` is set to `true`.

Rules:
1. Only flip `heldout_unlocked` after the final strategy configuration is approved.
2. Record every held-out evaluation below *before* looking at results.
3. After the first evaluation, the held-out set is spent. Any further change
   to strategies or parameters must be reported as post-held-out.

| Date (UTC) | Who approved | Configuration (commit hash) | Notes |
|---|---|---|---|
| _not yet evaluated_ | | | |
