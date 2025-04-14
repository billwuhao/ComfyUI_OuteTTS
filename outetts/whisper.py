import torch
import torchaudio
from loguru import logger
import whisper
from outetts.ttsv3.audio_processor import Features
from outetts.utils.preprocessing import text_normalizations
from outetts.audio_dac import DacInterface


class AudioProcessor:
    def __init__(self, device, audio, whisper_model, dac_model):
        self.device = device
        self.audio = audio
        self.features = Features(self.device)
        self.whisper_model = whisper_model
        self.audio_codec = DacInterface(dac_model, self.device)

    def clean(self):
        import gc
        self.audio_codec.clean()
        self.whisper_model = None
        self.audio_codec = None
        self.features = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def comfy_audio_to_audio(self):
        waveform, sample_rate = self.audio["waveform"], self.audio["sample_rate"]
        waveform = waveform.squeeze(0)
        waveform = waveform.to(self.device)
                
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(self.device)
            waveform = resampler(waveform)
                
        if len(waveform.shape) > 1:
            waveform = torch.mean(waveform, dim=0)

        return whisper.pad_or_trim(waveform.float()).to(self.device)

    def create_speaker_from_whisper(self):
        waveform, sample_rate = self.audio["waveform"], self.audio["sample_rate"]
        audio = self.audio_codec.convert_audio_tensor(waveform, sample_rate)
        seconds = audio.flatten().size(0) / self.audio_codec.sr
        if seconds > 20:
            raise ValueError("Speaker audio is longer than 20 seconds. Use a shorter clip for best results.")
        if seconds > 15:
            logger.warning("Speaker audio is longer than 15 seconds. For best results, consider using an audio clip up to 15 seconds.")

        data = self.whisper_model.transcribe(self.comfy_audio_to_audio(), 
                # initial_prompt="如果是中文,请严格使用简体中文:", 
                word_timestamps=True)
        text = text_normalizations(data['text'])
        words = []
        for s in data['segments']:
            words.extend([{'word': i['word'].strip(), 'start': float(i['start']), 'end': float(i['end'])} for i in s['words']])

        return self.create_speaker_from_dict({"audio": {"tensor": audio}, "text": text, "words": words})
    
    def create_speaker_from_dict(self, data: dict):
        audio = data['audio']['tensor'].to(self.device)
        full_codes = self.audio_codec.encode(audio, verbose=True).tolist()[0]

        c1 = full_codes[0]
        c2 = full_codes[1]

        sr = self.audio_codec.sr
        text = data['text']
        words = data['words']

        tps = 75
        audio = audio.squeeze(0)
        global_features = self.features.extract_audio_features(audio, sr)

        start = None
        word_codes = []
        max_extension = 20

        for idx, i in enumerate(words):
            if start is None:
                start = max(0, int(i['start'] * tps) - max_extension)
            word = i['word'].strip()
            if idx == len(words) - 1:
                end = min(len(c1), int(i['end'] * tps) + max_extension)
            else:
                end = int(i['end'] * tps)

            word_c1 = c1[start:end]
            word_c2 = c2[start:end]

            word_audio = audio[:, int(i['start']*sr):int(i['end']*sr)]
            features = self.features.extract_audio_features(word_audio, sr)

            start = end

            word_codes.append({
                "word": word,
                "duration": round(len(word_c1) / tps, 2),
                "c1": word_c1,
                "c2": word_c2,
                "features": features
            })

        return {
            "text": text,
            "words": word_codes,
            "global_features": global_features
        }
