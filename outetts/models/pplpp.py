import torch
from loguru import logger
from transformers import LogitsProcessor

# --------------------------------------------------------------------------------------------------------
# RepetitionPenaltyLogitsProcessor
# This patch implements a windowed repetition penalty for OuteTTS 1.0, 
# which is essential for maintaining high-quality speech synthesis output. 
# The standard HuggingFace implementation applies repetition penalties across the entire input context, 
# but OuteTTS requires a more focused approach that only considers the most recent tokens.
# --------------------------------------------------------------------------------------------------------

class RepetitionPenaltyLogitsProcessorPatch(LogitsProcessor):
    def __init__(self, penalty: float):
        penalty_last_n = 64
        logger.info(f"🔄 Using patched RepetitionPenaltyLogitsProcessor -> RepetitionPenaltyLogitsProcessorPatch | penalty_last_n: {penalty_last_n}")
        if penalty_last_n is not None:
            if not isinstance(penalty_last_n, int) or penalty_last_n < 0:
                raise ValueError(f"`penalty_last_n` has to be a non-negative integer, but is {penalty_last_n}")
        if not isinstance(penalty, float) or penalty <= 0:
            raise ValueError(f"`penalty` has to be a positive float, but is {penalty}")

        self.penalty_last_n = penalty_last_n
        self.penalty = penalty

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            input_ids (`torch.LongTensor`):
                Indices of input sequence tokens in the vocabulary (shape `(batch_size, sequence_length)`).
            scores (`torch.FloatTensor`):
                Prediction scores of a language modeling head (shape `(batch_size, vocab_size)`).

        Returns:
            `torch.FloatTensor`: The modified prediction scores.
        """
        # Check if penalties should be applied
        if self.penalty_last_n == 0 or self.penalty == 1.0:
            return scores

        batch_size, seq_len = input_ids.shape
        vocab_size = scores.shape[-1]

        # Process each batch item independently
        for b in range(batch_size):
            # 1. Determine the penalty window
            start_index = max(0, seq_len - self.penalty_last_n)
            window_indices = input_ids[b, start_index:] # Shape: (window_len,)

            if window_indices.numel() == 0: # Skip if window is empty
                continue

            # 2. Find unique tokens within the window
            tokens_in_window = set(window_indices.tolist())

            # 3. Apply repetition penalty to the scores for this batch item
            for token_id in tokens_in_window:
                if token_id >= vocab_size:
                    continue 

                logit = scores[b, token_id]

                if logit <= 0:
                    logit *= self.penalty
                else:
                    logit /= self.penalty

                # Update the score
                scores[b, token_id] = logit

        return scores
    