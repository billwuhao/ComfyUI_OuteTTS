import json
import torch
from loguru import logger
from dataclasses import dataclass, field
import folder_paths
import os
import dac
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from outetts.models import info
from outetts.models.hfmodel import HFModel
from outetts.utils.chunking import chunk_text
from outetts.utils import preprocessing
from outetts.audio_dac import DacInterface
from outetts.whisper import AudioProcessor
from outetts.ttsv3.prompt_processor import PromptProcessor


@dataclass
class SamplerConfig:
    temperature: float = 0.4
    repetition_penalty: float = 1.1
    repetition_range: int = 64
    # max_new_tokens: int = 100
    top_k: int = 40
    top_p: float = 0.9
    min_p: float = 0.05
    mirostat_tau: int = 5
    mirostat_eta: float = 0.1
    mirostat: bool = False

@dataclass
class GenerationConfig:
    text: str
    voice_characteristics: str = None
    speaker: dict = None
    generation_type: info.GenerationType = info.GenerationType.CHUNKED
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)
    max_length: int = 8192
    additional_gen_config: dict = field(default_factory=lambda: {})
    additional_dynamic_generator_config: dict = field(default_factory=lambda: {})


device = "cuda" if torch.cuda.is_available() else "cpu"

models_dir = folder_paths.models_dir
whisper_model_path = os.path.join(models_dir, "TTS", "whisper-large-v3-turbo", "large-v3-turbo.pt")
tts_model_path_v1 = os.path.join(models_dir, "TTS", "Llama-OuteTTS-1.0-1B")
tts_model_path_v2 = os.path.join(models_dir, "TTS", "OuteTTS-1.0-0.6B")
dac_model_path = os.path.join(models_dir, "TTS", "DAC.speech.v1.0", "weights_24khz_1.5kbps_v1.0.pth")
speakers_dir = os.path.join(models_dir, "TTS", "speakers", "Oute_Speakers")
os.makedirs(speakers_dir, exist_ok=True)

def get_compatible_dtype():
    """
    Returns the most compatible dtype for PyTorch based on the user's hardware:
    """
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            logger.info("BF16 support available. Using torch.bfloat16 data type.")
            return torch.bfloat16
        else:
            logger.info("BF16 support not available. Using torch.float16 data type.")
            return torch.float16
    else:
        logger.info("CUDA not available. Using torch.float32 data type.")
        return torch.float32

class OuteTTS:
    def __init__(self, device, tts_model, dac_model, tokenizer, config, genconfig):
        self.device = device
        self.prompt_processor = PromptProcessor(tokenizer)
        self.dac = DacInterface(dac_model, self.device)
        self.hfmodel = HFModel(tts_model)
        self.config = config
        self.genconfig = genconfig

    def clean(self):
        import gc
        self.prompt_processor.clean()
        self.hfmodel.clean()
        self.dac.clean()
        self.prompt_processor = None
        self.hfmodel = None
        self.dac = None
        gc.collect()
        torch.cuda.empty_cache()

    def _prepare_prompt(self, prompt: str):
        return self.prompt_processor.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)
    
    def prepare_prompt(self, text: str, speaker: dict = None):
        prompt = self.prompt_processor.get_completion_prompt(text, speaker)
        return self._prepare_prompt(prompt)
    
    def get_audio(self, tokens):
        output = self.prompt_processor.extract_audio_from_tokens(tokens)
        if not output:
            raise ValueError("No audio found in output.")

        return self.dac.decode(
            torch.tensor([output], dtype=torch.int64).to(self.dac.device)
        )

    def check_generation_max_length(self, max_length):
        if max_length is None:
            raise ValueError("max_length must be specified.")
        if max_length > self.config["max_seq_length"]:
            raise ValueError(f"Requested max_length ({max_length}) exceeds the current max_seq_length ({self.config["max_seq_length"]}).")

    def _generate(self, input_ids):
        output = self.hfmodel.generate(
            input_ids=input_ids,
            config=self.genconfig
        )
        return output[input_ids.size()[-1]:]
    
    def guided_words_generation(self):
        text_chunks = chunk_text(self.genconfig.text)
        chunk_size = len(text_chunks)
        word_token = self.prompt_processor.tokenizer.encode(
            self.prompt_processor.special_tokens.word_start, add_special_tokens=False
        )[0]
        end_token = self.prompt_processor.tokenizer.encode(
            self.prompt_processor.special_tokens.audio_end, add_special_tokens=False
        )[0]
        logger.info(f"Created: {chunk_size} text chunks")
        all_outputs = []

        def create_insert(word):
            insert = [word_token] 
            insert.extend(self.prompt_processor.tokenizer.encode(
                word + self.prompt_processor.special_tokens.features, add_special_tokens=False))
            return insert
        
        def cat_inputs(input_ids, output):
            if isinstance(input_ids, torch.Tensor):
                return torch.cat(
                    [input_ids, torch.tensor([output], dtype=torch.int64).to(input_ids.device)], dim=1
                )
            elif isinstance(input_ids, list):
                input_ids.extend(output)
                return input_ids
            elif isinstance(input_ids, str):
                return input_ids + self.prompt_processor.tokenizer.decode(output, skip_special_tokens=False)
            else:
                raise ValueError("Invalid input_ids type")

        for i, chunk in enumerate(text_chunks):
            logger.info(f"Proccessing: Chunk {i+1} / {chunk_size}")
            words = preprocessing.get_words(chunk)
            # Initialize first word
            first_word = words.pop(0)
            print(f"\nInserting first word: {repr(first_word)}")
            input_ids = self.prepare_prompt(chunk + first_word + self.prompt_processor.special_tokens.features, self.genconfig.speaker)
            output = []
            break_next = False
            while True:
                for token in self.hfmodel._generate_stream(input_ids, self.genconfig):
                    if token == word_token or token == end_token:
                        if not words:
                            break
                        insert = create_insert(words.pop(0))
                        output.extend(insert)
                        all_outputs.extend(insert)
                        print(f"\nInserting: {repr(self.prompt_processor.tokenizer.decode(insert, skip_special_tokens=False))}")
                        input_ids = cat_inputs(input_ids, output)
                        output = []
                        break
                    else:
                        output.append(token)
                        all_outputs.append(token)

                if not words:
                    if break_next:
                        break
                    break_next = True

        return all_outputs

    def chunk_generation(self):
        text_chunks = chunk_text(self.genconfig.text)
        audio_chunks = []
        chunk_size = len(text_chunks)

        logger.info(f"Created: {chunk_size} text chunks")
        for i, chunk in enumerate(text_chunks):
            logger.info(f"Proccessing: Chunk {i+1} / {chunk_size}")

            input_ids = self.prepare_prompt(chunk, self.genconfig.speaker)

            output = self._generate(input_ids)
            audio_chunks.extend(output)

        return audio_chunks
    
    def regular_generation(self):
        input_ids = self.prepare_prompt(self.genconfig.text, self.genconfig.speaker)
        return self._generate(input_ids)

    @torch.inference_mode()
    def generate(self):
        self.check_generation_max_length(self.genconfig.max_length)

        if self.genconfig.text.strip() == "":
            raise ValueError("text can not be empty!")
            
        if self.genconfig.generation_type == info.GenerationType.CHUNKED:
            output = self.chunk_generation()

        elif self.genconfig.generation_type == info.GenerationType.GUIDED_WORDS:
            logger.warning("Guided words generation is experimental and may not work as expected.")
            raise ValueError("Guided words generation is only supported for InterfaceVersion.V3")

        elif self.genconfig.generation_type == info.GenerationType.REGULAR:
            logger.info("Using regular generation, consider using chunked generation for long texts.")
            output = self.regular_generation()
        else:
            raise ValueError(f"Unsupported generation type: {self.genconfig.generation_type}")

        audio = self.get_audio(output)
        logger.success("Generation finished!")
        return (audio, self.dac.sr)


def get_speakers():
    if not os.path.exists(speakers_dir):
        os.makedirs(speakers_dir, exist_ok=True)
        return []
    
    speakers = [f.rsplit(".", 1)[0] for f in os.listdir(speakers_dir) if f.endswith('.json')]
    return speakers

TTS_MODEL = None
TOKENIZER = None
DAC_MODEL = None
WHISPER_MODEL = None
class OuteTTSRun:
    def __init__(self):
        self.dtype = get_compatible_dtype()
        self.device = device
        self.model_version = None

    @classmethod
    def INPUT_TYPES(s):
        speakers = get_speakers()
        # default_speaker = speakers[0] if speakers else "None"
        return {"required": {
                    "model": (["1B","0.6B"], {"default": "0.6B"}),
                    "text": ("STRING",),
                    },
                "optional": {
                    "audio": ("AUDIO",),
                    "speaker": (speakers + ["None"], {"default": "None"}),
                    # "max_new_tokens": ("INT", {"default": 500, "min": 1, "max": 8192}),
                    "save_speaker": ("BOOLEAN", {"default": True}),
                    "speaker_name": ("STRING", {"default": ""}),
                    "unload_model": ("BOOLEAN", {
                        "default": False,
                        "tooltip": "Unload model from memory after use"
                    }),
                    "chunked": ("BOOLEAN", {"default": True}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                }
            }

    CATEGORY = "🎤MW/MW-OuteTTS"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "gen"

    def save_speaker(self, speaker: dict, speaker_name):
        speaker['interface_version'] = 3
        file_path = os.path.join(speakers_dir, speaker_name + ".json")
        folder_path = os.path.dirname(file_path)
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(file_path, "w", encoding='utf-8') as f: 
            json.dump(speaker, f, ensure_ascii=False, indent=2)
        logger.info(f"Speaker saved to: {file_path}")

    def gen(self, model, text, speaker, 
            # max_new_tokens, 
            audio=None, unload_model=True, save_speaker=True, speaker_name="", chunked=True, seed=0):
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if save_speaker:
            if speaker_name.strip() == "":
                raise ValueError("save_speaker: Speaker name can not be empty!")

        genconfig = GenerationConfig(text="")
        speaker = None if speaker == "None" else speaker
        genconfig.text = text
        # genconfig.sampler_config.max_new_tokens = max_new_tokens
        
        config = {
            "verbose": False,
            "device": self.device,
            "dtype": self.dtype,
            "additional_model_config": {},
            "max_seq_length": 8192,
            "n_gpu_layers": 0,
        }

        try:
            from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
            config["additional_model_config"]["attn_implementation"] = "flash_attention_2"
            logger.success("Flash attention available. Using flash_attention_2 implementation.")
        except:
            logger.warning("Flash attention 2 not available. Using default attention implementation.\nFor faster inference on supported hardware, consider installing FlashAttention using:\npip install flash-attn --no-build-isolation")
        
        global TTS_MODEL, TOKENIZER, DAC_MODEL, WHISPER_MODEL
        if TTS_MODEL is None or TOKENIZER is None or DAC_MODEL is None or WHISPER_MODEL is None or self.model_version != model:
            WHISPER_MODEL = whisper.load_model(whisper_model_path).to(self.device)
            DAC_MODEL = dac.DAC.load(dac_model_path).to(self.device).eval()
            if model == "1B":
                tts_model_path = tts_model_path_v1
                self.model_version = "1B"
            else:
                tts_model_path = tts_model_path_v2
                self.model_version = "0.6B"

            TOKENIZER = AutoTokenizer.from_pretrained(tts_model_path)
            TTS_MODEL = AutoModelForCausalLM.from_pretrained(
                tts_model_path,
                torch_dtype=self.dtype,
                **config["additional_model_config"]
            ).to(self.device)

        if speaker is not None:
            file_path = os.path.join(speakers_dir, speaker + ".json")
            with open(file_path, "r", encoding='utf-8') as f:
                speaker = json.load(f)
        elif audio is not None:
            self.audio_processor = AudioProcessor(self.device, audio, WHISPER_MODEL, DAC_MODEL)
            speaker = self.audio_processor.create_speaker_from_whisper()
            self.save_speaker(speaker, speaker_name.strip())
        else:
            raise ValueError("No speaker or audio provided!")

        genconfig.speaker = speaker

        if not chunked:
            genconfig.generation_type = info.GenerationType.REGULAR

        otts = OuteTTS(self.device, TTS_MODEL, DAC_MODEL, TOKENIZER, config, genconfig)
        audio_array, sample_rate = otts.generate()

        if unload_model:
            otts.clean()
            self.audio_processor.clean()
            self.audio_processor = None
            TTS_MODEL = None
            TOKENIZER = None
            DAC_MODEL = None
            WHISPER_MODEL = None
            otts = None
            torch.cuda.empty_cache()

        # Move data back to CPU before return
        return ({"waveform": audio_array.cpu(), "sample_rate": sample_rate},)



NODE_CLASS_MAPPINGS = {
    "OuteTTSRun": OuteTTSRun,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OuteTTSRun": "OuteTTS Run",
}