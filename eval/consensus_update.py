"""Consensus-reinforced correction: does centroid merging add anything to
recency's edge once the corrected value is restated more than once?

This dataset is deliberately built to favour SR-KV, following the same
disclosed-bias discipline gist_mcq's H3 already uses ("this task is synthetic
and designed specifically to give clustering the conditions it should need" -
PREREGISTRATION.md). It is not a claim that SR-KV wins on realistic text; it
is the sharpest test this project can construct for one specific, falsifiable
idea, and the result is reported either way.

Where this idea comes from (both already committed, see
results/conversation_pilot_v1 and results/incident_stream_v1):
`recency_hard_evict` consistently beat `snapkv_unified` whenever the correct
answer was the *most recent* version of a fact that had been corrected
mid-passage - a clean, reproducible pattern. But `sr_kv` (recency AND
clustering together) never cleanly beat `recency_hard_evict` itself in either
pilot; at one budget in conversation_pilot_v1 it was *worse* than
recency-only. Clustering added nothing there, which makes sense: neither
pilot ever restated the corrected fact more than once, so there was nothing
redundant for a centroid to average.

H3b (gist_mcq, already in this repo) established the one structural reason
clustering should beat hard top-k selection: "a centroid pulled from five
similar mentions should point more strongly toward that topic than one
pulled from a single mention... hard eviction has no equivalent of how many
almost-evicted tokens agreed." This dataset combines that with the recency
structure that already worked: the corrected destination is reinforced
`n_reinforcements` times (1, 3, or 5) late in the passage, competing against
a single, equally-late, individually salient DECOY restating the superseded
destination.

Mechanism prediction. `recency_hard_evict` (and `snapkv_unified`) can only
keep a fixed number of individually top-scoring tokens; if the decoy
out-scores any one reinforcement on its own, a hard-eviction method can keep
the decoy over a weaker-but-correct mention. `sr_kv`'s clustering step
instead folds whichever reinforcements do not win an individual slot into an
attention-weighted centroid - several consistent correct mentions pulling one
summary vector toward the right answer, rather than each competing alone and
risking individual loss to the decoy. At `n_reinforcements=1` there is
nothing to cluster, so sr_kv should collapse to being indistinguishable from
recency_hard_evict (this is the built-in control, not a separate dataset);
the real prediction is a **dose-response** in `n_reinforcements`, not a flat
offset - sr_kv's edge over recency_hard_evict should grow from n=1 to n=5.

Two-pass measure and MCQ scoring are reused unmodified from `eval.gist_mcq`,
same discipline as `eval.conversation_pilot`.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass

from eval.gist_mcq import GistSample, _make_context, score  # noqa: F401 (score re-exported)

VERSION = "consensus-update-v1"

#: variant name -> how many times the corrected value is reinforced late in
#: the passage. n=1 is the built-in control (reduces to a single-mention
#: correction, structurally close to conversation_pilot's current_request);
#: n=3/5 are the dose-response arms this dataset exists to measure.
N_REINFORCEMENTS = {"consensus_n1": 1, "consensus_n3": 3, "consensus_n5": 5}
VARIANTS = tuple(N_REINFORCEMENTS)

PRODUCTS = ("keyboard", "router", "headphones", "monitor", "printer", "battery pack")
#: exactly four, so {old, new} leaves exactly two left over as the unrelated
#: MCQ distractors - no sampling edge case to guard.
LOCATIONS = ("home address", "office reception", "local collection point", "registered workshop")

FILLER = (
    "Agent: I am opening the service notes for the {product}. Customer: Thank you for checking.",
    "Customer: Could you explain the {topic} process? Agent: I will describe the standard steps.",
    "Agent: The {topic} information is in the help centre. Customer: I can look at that later.",
    "Customer: I have been reading the troubleshooting guide. Agent: The diagrams can be useful.",
    "Agent: Let me check the account screen. Customer: I can wait while you do that.",
    "Customer: The last explanation was clear. Agent: Please ask if another point needs clarification.",
    "Agent: I am reviewing the {topic} procedure. Customer: It helps to understand how it works.",
    "Customer: Is there documentation about {topic}? Agent: The support portal has a general guide.",
    "Agent: I have returned to the service screen. Customer: We can continue with the discussion.",
    "Customer: I want to understand the available support resources. Agent: There are written guides and tutorials.",
    "Agent: The {product} manual has an index of common questions. Customer: That sounds helpful.",
    "Customer: I was unsure about some terminology. Agent: I can explain the terms used in the guide.",
)
TOPICS = ("warranty", "troubleshooting", "recycling", "software setup", "cleaning",
          "technical support", "accessibility", "maintenance", "registration", "diagnostics")

_INITIAL_TMPL = (
    "Customer: I need a replacement {product}. Please send it to my {old}.\n"
    "Agent: I have recorded that request."
)
_CORRECTION_TMPL = (
    "Customer: I need to correct the delivery destination. Use my {new} instead of my {old}.\n"
    "Agent: Confirmed, I have updated the destination."
)
#: varied phrasing per reinforcement on purpose - token-identical repeats
#: would let the cache dedupe trivially and never exercise the clustering
#: path on genuinely distinct candidate slots.
_REINFORCEMENT_TMPLS = (
    "Agent: Just to confirm once more, the {product} is going to your {new}.",
    "Agent: Noting again for the record: the delivery destination is your {new}.",
    "Agent: Confirming the destination remains your {new}, as updated.",
    "Agent: For this ticket, the shipment is still set for your {new}.",
    "Agent: Reconfirming before dispatch: we have your {new} on file.",
)
#: one single, individually salient mention of the superseded address.
#: Deliberately framed as a past, explicitly-superseded fact ("before it was
#: corrected") rather than a claimed override - an earlier version asserted
#: the old address was current again ("disregard the update... after all"),
#: which made the task genuinely ambiguous even for the uncompressed `full`
#: reference (sanity run 2026-10-13: full scored 6/15 wrong, and every single
#: wrong answer picked the decoy, zero picked an unrelated option - a sanity
#: gate failure, not a caching effect). This version still puts the old
#: address's string back into the late, budget-contested region for a hard-
#: eviction policy's windowed-attention scoring to possibly misjudge, without
#: asking any reader - compressed or not - to resolve a contradiction.
_DECOY_TMPL = (
    "Agent: For the record, I see a note that the original request had "
    "listed your {old} before it was corrected."
)

#: reinforcements land across the back third of the passage; the decoy sits
#: among them (depth 87, between the 4th and 5th reinforcement slots) so it
#: genuinely competes for budget rather than simply arriving last.
_REINFORCEMENT_DEPTHS_ALL = (65, 72, 78, 84, 90)
_DECOY_DEPTH = 87
_CORRECTION_DEPTH = 60


@dataclass
class ConsensusSample(GistSample):
    context_sha256: str = ""
    dataset_version: str = VERSION
    seed: int = 0
    old_destination: str = ""
    new_destination: str = ""
    decoy_destination: str = ""
    n_reinforcements: int = 1
    reinforcement_depths: tuple = ()
    decoy_depth: int = 0

    @property
    def task_id(self) -> str:
        return (f"consensus/{VERSION}/seed{self.seed}/ctx{self.context_len}"
                f"/s{self.sample_idx}/{self.variant}")

    def to_dict(self):
        return asdict(self)


def build_samples(tokenizer, *, context_len=4096, n_samples=50, seed=270901,
                   variants=VARIANTS):
    """One (variant, sample_idx) cell per combination.

    Same per-conversation independent rng discipline as
    `eval.conversation_pilot.build_samples`: the filler/location/product draw
    for a given `sample_idx` never changes when `n_samples` or `variants`
    changes, so partial runs and resumes stay aligned, and every variant for
    a given conversation shares the identical context except for how many
    reinforcement sentences are spliced in.
    """
    for v in variants:
        if v not in N_REINFORCEMENTS:
            raise ValueError(f"unknown variant {v!r}; known: {sorted(N_REINFORCEMENTS)}")
    if context_len < 1024:
        raise ValueError("Use at least 1024 tokens so every reinforcement fits distinctly.")

    samples = []
    for idx in range(n_samples):
        rng = random.Random(f"{VERSION}:{seed}:{idx}")
        product = rng.choice(PRODUCTS)
        old, new = rng.sample(LOCATIONS, 2)
        unrelated_a, unrelated_b = [l for l in LOCATIONS if l not in (old, new)]

        filler = []
        for _ in range(context_len // 6 + 150):
            filler.append(rng.choice(FILLER).format(product=product, topic=rng.choice(TOPICS)))
        hay_ids = tokenizer("\n".join(filler), add_special_tokens=False)["input_ids"]

        for variant in variants:
            n_reinf = N_REINFORCEMENTS[variant]
            chosen_tmpls = rng.sample(_REINFORCEMENT_TMPLS, n_reinf)
            chosen_depths = _REINFORCEMENT_DEPTHS_ALL[:n_reinf]

            fragments = [
                (0, f"\n{_INITIAL_TMPL.format(product=product, old=old)}\n"),
                (_CORRECTION_DEPTH, f"\n{_CORRECTION_TMPL.format(product=product, new=new, old=old)}\n"),
                (_DECOY_DEPTH, f"\n{_DECOY_TMPL.format(product=product, old=old)}\n"),
            ]
            for depth, tmpl in zip(chosen_depths, chosen_tmpls):
                fragments.append((depth, f"\n{tmpl.format(product=product, new=new)}\n"))

            context = _make_context(tokenizer, hay_ids, context_len, fragments)
            digest = hashlib.sha256(context.encode()).hexdigest()

            options = [new, old, unrelated_a, unrelated_b]
            rng.shuffle(options)
            letter = "ABCD"[options.index(new)]
            question = (
                "Where should the replacement currently be delivered, according to the "
                "most recent confirmed instruction? Answer with only the letter of the "
                "correct option.\n"
                + "\n".join(f"{l}) {o}" for l, o in zip("ABCD", options)) + "\nAnswer:"
            )
            samples.append(ConsensusSample(
                context_len=context_len, variant=variant, sample_idx=idx,
                options=options, answer_letter=letter, context=context, question=question,
                max_new_tokens=12, context_sha256=digest, seed=seed,
                old_destination=old, new_destination=new, decoy_destination=old,
                n_reinforcements=n_reinf, reinforcement_depths=tuple(chosen_depths),
                decoy_depth=_DECOY_DEPTH,
            ))
    return samples
