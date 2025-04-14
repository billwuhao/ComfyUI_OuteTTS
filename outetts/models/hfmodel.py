import transformers.generation.utils as generation_utils
from transformers import AutoModelForCausalLM
import torch

from outetts.models.pplpp import RepetitionPenaltyLogitsProcessorPatch
from outetts.models.info import GenerationType


generation_utils.RepetitionPenaltyLogitsProcessor = RepetitionPenaltyLogitsProcessorPatch
AutoModelForCausalLM.generate = generation_utils.GenerationMixin.generate

class HFModel:
    def __init__(
        self,
        model,
    ) -> None:
        self.model = model

    def _generate(self, input_ids: torch.Tensor, config):
        if config.sampler_config.temperature > 0:
            config.additional_gen_config["do_sample"] = True
        return self.model.generate(
            input_ids,
            max_length=config.max_length,
            temperature=config.sampler_config.temperature,
            repetition_penalty=config.sampler_config.repetition_penalty,
            top_k=config.sampler_config.top_k,
            top_p=config.sampler_config.top_p,
            min_p=config.sampler_config.min_p,
            **config.additional_gen_config,
        )[0].tolist()
    
    def _generate_stream(self, input_ids: torch.Tensor, config):
        raise NotImplementedError("Stream generation is not supported for HF models.")

    def generate(self, input_ids: list[int], config):
        if config.generation_type == GenerationType.STREAM:
            return self._generate_stream(input_ids, config)
        return self._generate(input_ids, config)
    
    def clean(self):
        import gc
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
