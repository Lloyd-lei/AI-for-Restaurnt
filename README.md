# OpenAI Realtime API 语音对话客户端

一个功能完整的 OpenAI Realtime API 客户端，支持实时语音对话、打断、回音消除和音频预处理。

## ✨ 功能特点

- 🎙️ **实时语音识别（ASR）**：使用 OpenAI 内置的 Whisper 实时版本
- 🧠 **智能对话（LLM）**：基于 GPT-4o 实时模型
- 🗣️ **语音合成（TTS）**：shimmer 声音（温暖友好的女声）
- ⚡ **支持打断（Barge-in）**：在 AI 说话时可以随时打断
- 🔇 **回音消除**：智能过滤 AI 语音，防止 ASR 识别回声
- 🌍 **多语言支持**：自动检测并切换中文/英语/日语等
- 🎛️ **音频预处理**：降噪、增益控制
- ⚡ **可调播放速度**：支持 1.0x - 1.5x 倍速播放

## 🚀 快速开始

### 1. 安装依赖

#### macOS 用户

```bash
# 安装 PortAudio
brew install portaudio

# 安装 Python 依赖
pip install -r requirements.txt
```

#### Linux 用户

```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev

# 安装 Python 依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件，添加你的 OpenAI API Key：

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

⚠️ **重要**：请确保 `.env` 文件不会被提交到 Git（已在 `.gitignore` 中）

### 3. 运行程序

```bash
python3 realtime_local_asr.py
```

## ⚙️ 配置参数

在 `realtime_client.py` 文件顶部可以调整以下参数：

### 播放速度

```python
PLAYBACK_SPEED = 1.2  # 1.0 = 正常，1.2 = 1.2倍速
```

### 回音消除

```python
INTERRUPT_THRESHOLD = 100  # 打断阈值（50-500）
ENABLE_ECHO_CANCELLATION = True  # 是否启用回音消除
```

### 音频预处理

```python
ENABLE_GAIN_CONTROL = True  # 增益控制（放大音量）
TARGET_GAIN = 1.5  # 增益倍数（1.0-3.0）
ENABLE_NOISE_GATE = True  # 噪声门（过滤低能量噪音）
NOISE_GATE_THRESHOLD = 30  # 噪声门阈值
```

### 声音选择

```python
"voice": "shimmer"  # 可选: alloy, echo, fable, onyx, nova, shimmer
```

## 💡 使用建议

1. **使用耳机**：最佳体验，避免回音问题
2. **环境噪音大**：提高 `NOISE_GATE_THRESHOLD` 到 50-100
3. **麦克风音量小**：提高 `TARGET_GAIN` 到 2.0-3.0
4. **打断太敏感**：提高 `INTERRUPT_THRESHOLD` 到 200-300
5. **语速慢**：提高 `PLAYBACK_SPEED` 到 1.3-1.5

## 📋 技术细节

### 模块运行位置

| 模块        | 运行位置      | 说明                 |
| ----------- | ------------- | -------------------- |
| 🎙️ ASR      | OpenAI 服务器 | Whisper-1 实时版     |
| 🧠 LLM      | OpenAI 服务器 | GPT-4o 实时版        |
| 🗣️ TTS      | OpenAI 服务器 | shimmer 声音         |
| 🎛️ VAD      | OpenAI 服务器 | 服务器端语音活动检测 |
| 🖥️ 音频 I/O | 本地          | 采集、预处理、播放   |

### 本地音频预处理

1. **采集音频** → 麦克风捕获 PCM16 音频
2. **噪声门** → 过滤低能量环境噪音
3. **增益控制** → 放大音量，增强清晰度
4. **Base64 编码** → 准备发送
5. **发送到服务器** → WebSocket 传输

## 🎯 功能说明

### 1. 回音消除机制

- AI 播放时暂停发送麦克风数据
- 实时计算音频能量
- 检测到用户说话时触发打断

### 2. 多语言自动切换

- 服务器端 ASR 自动识别语言
- LLM 根据识别结果选择回复语言
- 无需手动切换

### 3. 音频预处理

- **降噪**：过滤静音和低能量噪音
- **增益**：智能放大，防止削波

## 🐛 常见问题

**Q: 回音问题？**
A: 使用耳机最有效，或调高 `NOISE_GATE_THRESHOLD`

**Q: 打断不灵敏？**
A: 降低 `INTERRUPT_THRESHOLD` 到 50-100

**Q: 声音太小？**
A: 提高 `TARGET_GAIN` 到 2.0-3.0

**Q: 安装 pyaudio 失败？**
A: 先安装 PortAudio (`brew install portaudio`)

## 📄 许可证

MIT License
