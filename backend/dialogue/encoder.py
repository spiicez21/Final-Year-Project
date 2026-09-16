"""Sentence vectors from the encoder inside the player-fact GLiNER.

Used by the intent classifier (intent.py) and by its training script
(training/extractor/train_intent.py), so a line is embedded identically in
both. Reusing the extractor's encoder means intent costs one extra forward
pass (~20 ms on CPU), not a second model in memory.
"""

from pathlib import Path

import torch


class SentenceEncoder:
    """Mean-pooled last hidden states of GLiNER's DeBERTa encoder."""

    def __init__(self, model_dir: Path | str, gliner_model=None):
        if gliner_model is None:
            from gliner import GLiNER
            gliner_model = GLiNER.from_pretrained(str(model_dir), local_files_only=True)
        self.encoder = gliner_model.model.token_rep_layer.bert_layer.model
        self.tokenizer = gliner_model.data_processor.transformer_tokenizer
        self.encoder.eval()

    @torch.no_grad()
    def encode(self, texts: list, batch_size: int = 64) -> torch.Tensor:
        device = next(self.encoder.parameters()).device
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer([t.lower() for t in texts[i:i + batch_size]], padding=True,
                                   truncation=True, max_length=64, return_tensors="pt").to(device)
            hidden = self.encoder(input_ids=batch["input_ids"],
                                  attention_mask=batch["attention_mask"]).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            out.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu())
        return torch.cat(out)
