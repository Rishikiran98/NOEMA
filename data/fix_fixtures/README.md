# Fix-search fixtures (M3.5)

Small repositories, each with one deliberate bug that its own tests catch. They are inputs for
`python -m cpu_platform.fix_search`. Reference fixes live only in `tests/test_m35_fix_search.py`,
never here, so a model cannot find the answer in the repository copy it is shown.

| Fixture | Bug |
|---|---|
| `inventory_offbyone` | `<` where `<=` is meant (boundary) |
| `pricing_wrong_constant` | VAT 0.02 instead of 0.20 |
| `text_dropped_condition` | truncation can leave a trailing `-` (missing strip) |
