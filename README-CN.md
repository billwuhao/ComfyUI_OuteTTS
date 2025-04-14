[中文](README-CN.md)|[English](README.md)

# QuteTTS 的 ComfyUI 节点

![](https://github.com/billwuhao/ComfyUI_OuteTTS/blob/main/images/2025-04-14_15-13-37.png)

文本转语音, 声音克隆(原音频最长不超过20秒), 自动保存说话者. 

干净, 洪亮, 清晰的人声克隆效果更好.

## 📣 更新

[2025-04-14]⚒️: 发布 v1.0.0。

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_OuteTTS.git
cd ComfyUI_OuteTTS
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 模型下载

- [Llama-OuteTTS-1.0-1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B): 下载放到 `ComfyUI/models/TTS` 目录下.
- [weights_24khz_1.5kbps_v1.0.pth](https://huggingface.co/ibm-research/DAC.speech.v1.0/blob/main/weights_24khz_1.5kbps_v1.0.pth): 下载放到 `ComfyUI/models/TTS/DAC.speech.v1.0` 目录下.
- [whisper-large-v3-turbo](https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt): 下载放到 `ComfyUI/models/TTS/whisper-large-v3-turbo` 目录下.

## 鸣谢

[OuteTTS](https://github.com/edwko/OuteTTS)