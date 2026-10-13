# Rejected sanity run (2026-10-13) — decoy wording was ambiguous even for `full`

Kept for transparency, same discipline as `incident_stream_v1`'s own sanity
failure. **Not used in any later comparison.**

The decoy sentence in this version of `eval/consensus_update.py` read:

> "Agent: URGENT CORRECTION - disregard the update, the {product} should go
> to your {old} after all."

This explicitly claimed to override the real correction, which is genuinely
ambiguous even for the uncompressed `full` reference, not just for a
compressed cache. Result: `full_accuracy = 0.60` (required >= 0.90), gate
failed as designed. Every one of the 6 wrong answers (out of 15) picked the
decoy/old destination — zero picked an unrelated option — which is what
showed this was a wording defect, not a capability limit:

    consensus_n1  0.60
    consensus_n3  0.40
    consensus_n5  0.80

Fixed by rewording the decoy to describe the old address only as an
explicitly superseded past fact ("before it was corrected"), never claiming
current validity. See the current `_DECOY_TMPL` in `eval/consensus_update.py`
and `tests/test_consensus_update.py::test_decoy_never_claims_the_old_address_is_still_current`,
a regression test added so this wording can't silently reappear.
