"""Deterministic synthetic support transcripts for the two-question pilot.

This is a controlled mechanism test, not a realistic-dialogue benchmark.
Every conversation has a corrected destination and an unchanged delivery
constraint. Both questions see exactly the same transcript. No corpus or
external language model is needed to construct the examples.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass

from eval.gist_mcq import GistSample, _make_context, score

VERSION = "conversation-pilot-v1"
VARIANTS = ("current_request", "old_constraint")
LOCATIONS = ("home address", "office reception", "local collection point", "registered workshop")
CONSTRAINTS = (
    "A signature is required when the parcel arrives.",
    "The parcel must not be left with a neighbour.",
    "The delivery must arrive before Friday.",
    "The parcel must arrive before noon.",
    "The courier must telephone before arriving.",
    "The parcel must remain in its original packaging.",
)
PRODUCTS = ("keyboard", "router", "headphones", "monitor", "printer", "battery pack")
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


@dataclass
class ConversationSample(GistSample):
    context_sha256: str = ""
    dataset_version: str = VERSION
    seed: int = 0
    old_destination: str = ""
    current_destination: str = ""
    persistent_constraint: str = ""
    correction_depth: int = 0
    constraint_depth: int = 0

    @property
    def task_id(self):
        return f"conversation/{VERSION}/seed{self.seed}/ctx{self.context_len}/s{self.sample_idx}/{self.variant}"

    def to_dict(self):
        return asdict(self)


def build_samples(tokenizer, *, context_len=4096, n_samples=50, seed=260927):
    if context_len < 768:
        raise ValueError("Use at least 768 tokens so all events fit.")
    samples = []
    for idx in range(n_samples):
        # Independent per-conversation randomness: prefixes and resume never
        # change when the requested sample count changes.
        rng = random.Random(f"{VERSION}:{seed}:{idx}")
        product = rng.choice(PRODUCTS)
        old, new = rng.sample(LOCATIONS, 2)
        constraint = rng.choice(CONSTRAINTS)
        correction_depth = (60, 75, 90)[idx % 3]
        constraint_depth = (5, 15, 25)[(idx // 3) % 3]
        filler = []
        # Generate more filler than needed once; the existing splice helper
        # truncates only filler, never the planted events.
        for _ in range(context_len // 8 + 100):
            filler.append(rng.choice(FILLER).format(product=product, topic=rng.choice(TOPICS)))
        hay_ids = tokenizer("\n".join(filler), add_special_tokens=False)["input_ids"]
        fragments = [
            (0, f"\nCustomer: I need a replacement {product}. Please send it to my {old}.\nAgent: I have recorded that request.\n"),
            (constraint_depth, f"\nCustomer: Please record this delivery requirement: {constraint}\nAgent: I have recorded your requirement for this shipment.\n"),
            (correction_depth, f"\nCustomer: I need to correct the delivery destination. Use my {new} instead of my {old}. All other requirements remain unchanged.\nAgent: Confirmed, I have updated the destination.\n"),
        ]
        context = _make_context(tokenizer, hay_ids, context_len, fragments)
        digest = hashlib.sha256(context.encode()).hexdigest()
        for variant in VARIANTS:
            if variant == "current_request":
                answer = new
                options = list(LOCATIONS)
                question = "Where should the replacement be delivered according to the customer's latest instruction?"
            else:
                answer = constraint
                options = [constraint] + rng.sample([c for c in CONSTRAINTS if c != constraint], 3)
                question = "Which delivery requirement from earlier in the conversation still applies?"
            rng.shuffle(options)
            letter = "ABCD"[options.index(answer)]
            question += " Answer with only the letter of the correct option.\n"
            question += "\n".join(f"{l}) {o}" for l, o in zip("ABCD", options)) + "\nAnswer:"
            samples.append(ConversationSample(
                context_len=context_len, variant=variant, sample_idx=idx,
                options=options, answer_letter=letter, context=context, question=question,
                max_new_tokens=12, context_sha256=digest, seed=seed,
                old_destination=old, current_destination=new,
                persistent_constraint=constraint, correction_depth=correction_depth,
                constraint_depth=constraint_depth,
            ))
    return samples
