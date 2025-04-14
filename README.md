[中文](README-CN.md)|[English](README.md)

# QuteTTS ComfyUI Node

![](https://github.com/billwuhao/ComfyUI_OuteTTS/blob/main/images/2025-04-14_15-13-37.png)

Text-to-speech, voice cloning (source audio up to 20 seconds), auto-saving speakers.

Clean, loud, and clear voice cloning works best.

## 📣 Updates

[2025-04-14] ⚒️: Released v1.0.0.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_OuteTTS.git
cd ComfyUI_OuteTTS
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Model Downloads

- [Llama-OuteTTS-1.0-1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B): Download and place in the `ComfyUI/models/TTS` directory.
- [weights_24khz_1.5kbps_v1.0.pth](https://huggingface.co/ibm-research/DAC.speech.v1.0/blob/main/weights_24khz_1.5kbps_v1.0.pth): Download and place in the `ComfyUI/models/TTS/DAC.speech.v1.0` directory.
- [whisper-large-v3-turbo](https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt): Download and place in the `ComfyUI/models/TTS/whisper-large-v3-turbo` directory.

## Acknowledgements

[OuteTTS](https://github.com/edwko/OuteTTS)