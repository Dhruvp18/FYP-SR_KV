"""Post-sanity diagnostic: replay identical tokens with native full-cache SDPA.

This is a harness diagnostic, not a replacement or retuning of pilot scores.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.incident_stream import build_streams, predicted_class, prompt_parts
from src.models import resolve_model_id


def main():
    torch.set_num_threads(2)
    model_id = resolve_model_id("qwen2.5-1.5b")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
    prefix, suffix = prompt_parts(tokenizer)
    out = ROOT / "results/incident_stream_v1"
    pilot = {}
    for path in out.glob("sanity_part*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["method"] == "full":
                pilot[row["stream_id"], row["checkpoint"]] = row["predicted"]
    records = []
    for stream in build_streams(n=4, seed=31003):
        ids = []
        for checkpoint, segment in enumerate(stream.segments):
            ids += tokenizer((prefix if checkpoint == 0 else "") + segment,
                             add_special_tokens=False)["input_ids"]
            query = ids + tokenizer(suffix, add_special_tokens=False)["input_ids"]
            inputs = torch.tensor([query], device="cuda")
            with torch.no_grad():
                generated = model.generate(
                    inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                    max_new_tokens=6, temperature=None, top_p=None, top_k=None,
                    pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(generated[0, len(query):], skip_special_tokens=True)
            pred = predicted_class(text)
            row = dict(stream_id=stream.stream_id, checkpoint=checkpoint,
                       expected=stream.expected[checkpoint], predicted=pred,
                       generated_text=text, incremental_predicted=pilot[stream.stream_id, checkpoint])
            records.append(row)
            print(json.dumps(row), flush=True)
    result = dict(records=records, correct=sum(r["predicted"] == r["expected"] for r in records),
                  agrees_with_incremental=sum(r["predicted"] == r["incremental_predicted"] for r in records),
                  n=len(records), scope="Post-sanity native SDPA harness diagnostic")
    (out / "full_native_diagnostic.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
