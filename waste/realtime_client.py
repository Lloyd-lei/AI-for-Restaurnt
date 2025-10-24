#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI Realtime API 客户端 - 支持语音打断功能
"""

import asyncio
import websockets
import json
import pyaudio
import base64
import os
from dotenv import load_dotenv
import signal
import sys
import struct
import math

# 加载环境变量
load_dotenv()

# 配置参数
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_API_URL = "wss://api.openai.com/v1/realtime"
MODEL = "gpt-4o-realtime-preview-2024-10-01"

# 音频参数（OpenAI Realtime API 要求）
SAMPLE_RATE = 24000  # 24kHz
CHANNELS = 1  # 单声道（OpenAI 要求）
CHUNK_SIZE = 4800  # 200ms 的音频块 (24000 * 0.2)
FORMAT = pyaudio.paInt16  # 16-bit PCM

# 回音消除参数
INTERRUPT_THRESHOLD = 0  # 打断阈值（音频能量），越大越不容易触发打断
ENABLE_ECHO_CANCELLATION = False  # 是否启用回音消除（AI播放时不发送麦克风数据）

# 播放速度参数
PLAYBACK_SPEED = 1  # 播放速度倍数（1.0 = 正常，1.2 = 1.2倍速，建议范围 1.0-1.5）

# 音频预处理参数
ENABLE_GAIN_CONTROL = False  # 是否启用增益控制
TARGET_GAIN = 1  # 目标增益倍数（建议 1.0-3.0）
ENABLE_NOISE_GATE = False  # 是否启用噪声门（过滤低能量噪音）
NOISE_GATE_THRESHOLD = 30  # 噪声门阈值，低于此值的音频会被静音

def calculate_audio_energy(audio_data):
    """计算音频能量（RMS - Root Mean Square）"""
    # 将字节数据转换为 16-bit 整数
    samples = struct.unpack(f'{len(audio_data)//2}h', audio_data)
    # 计算 RMS
    sum_squares = sum(sample ** 2 for sample in samples)
    rms = math.sqrt(sum_squares / len(samples))
    return int(rms)

def apply_gain(audio_data, gain=1.5):
    """应用增益（放大音频）"""
    samples = struct.unpack(f'{len(audio_data)//2}h', audio_data)
    # 应用增益并限制在 int16 范围内
    amplified = [max(-32768, min(32767, int(sample * gain))) for sample in samples]
    return struct.pack(f'{len(amplified)}h', *amplified)

def apply_noise_gate(audio_data, threshold=30):
    """应用噪声门（过滤低能量音频）"""
    energy = calculate_audio_energy(audio_data)
    if energy < threshold:
        # 返回静音
        return b'\x00' * len(audio_data)
    return audio_data

def preprocess_audio(audio_data):
    """
    音频预处理主函数
    包括：噪声门、增益控制等
    """
    processed = audio_data
    
    # 1. 噪声门（过滤环境噪音）
    if ENABLE_NOISE_GATE:
        processed = apply_noise_gate(processed, NOISE_GATE_THRESHOLD)
    
    # 2. 增益控制（增强音量）
    if ENABLE_GAIN_CONTROL:
        processed = apply_gain(processed, TARGET_GAIN)
    
    return processed

class RealtimeClient:
    def __init__(self):
        self.ws = None
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.is_running = False
        self.is_ai_speaking = False
        self.session_id = None
        self.cancel_sent = False  # 防止重复发送取消消息
        
    async def connect(self):
        """连接到 OpenAI Realtime API"""
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        url = f"{REALTIME_API_URL}?model={MODEL}"
        
        print("🔄 正在连接到 OpenAI Realtime API...")
        self.ws = await websockets.connect(url, extra_headers=headers)
        print("✅ 已连接到 OpenAI Realtime API")
        
        # 配置会话
        await self.configure_session()
        
    async def configure_session(self):
        """配置会话参数"""
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": (
                    "你是一个友好的多语言AI助手。请遵循以下规则：\n"
                    "1. 自动检测用户使用的语言\n"
                    "2. 用户说什么语言，你就用什么语言回复\n"
                    "3. 如果用户说中文，你就用中文回复\n"
                    "4. 如果用户说英语，你就用英语回复\n"
                    "5. 如果用户说日语，你就用日语回复\n"
                    "6. 保持简洁、自然的对话风格\n"
                    "7. 不要混用多种语言，始终使用用户当前使用的语言"
                ),
                "voice": "shimmer",  # shimmer 声音（女声，温暖友好）
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",  # 服务器端语音活动检测
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500  # 500ms 静音后结束
                }
            }
        }
        
        await self.ws.send(json.dumps(config))
        print("⚙️  会话配置已发送（多语言模式 + shimmer 声音）")
        
    async def start_audio_input(self):
        """启动麦克风音频输入"""
        self.input_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print("🎙️  麦克风已启动，开始监听...")
        
        while self.is_running:
            try:
                # 在线程中读取音频数据（避免阻塞事件循环）
                audio_data = await asyncio.to_thread(
                    self.input_stream.read, CHUNK_SIZE, False
                )
                
                # 检查 WebSocket 连接状态
                if not self.ws or self.ws.closed:
                    print("❌ WebSocket 连接已关闭")
                    break
                
                # 🔇 如果 AI 正在播放语音，不发送麦克风数据（防止回音）
                if self.is_ai_speaking and ENABLE_ECHO_CANCELLATION:
                    # 检查音频能量，判断用户是否在说话（简单的 VAD）
                    audio_level = calculate_audio_energy(audio_data)
                    
                    # 如果音频能量超过阈值，说明用户在说话，触发打断
                    if audio_level > INTERRUPT_THRESHOLD and not self.cancel_sent:
                        print(f"⚡ 检测到用户说话（能量: {audio_level}），打断AI回复")
                        # 发送取消响应的消息
                        try:
                            await self.ws.send(json.dumps({
                                "type": "response.cancel"
                            }))
                            self.cancel_sent = True
                            self.is_ai_speaking = False
                        except Exception as e:
                            print(f"❌ 发送取消消息失败: {e}")
                    else:
                        # AI 在说话且用户没有打断，跳过这帧（防止回音）
                        continue
                
                # 🎛️ 音频预处理（降噪、增益）
                processed_audio = preprocess_audio(audio_data)
                
                # 将音频编码为 base64 并发送
                audio_b64 = base64.b64encode(processed_audio).decode('utf-8')
                
                message = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }
                
                try:
                    await self.ws.send(json.dumps(message))
                except Exception as e:
                    if self.is_running:
                        print(f"❌ 发送音频数据失败: {e}")
                        break
                
                # 短暂延迟，避免发送过快
                await asyncio.sleep(0.01)
                
            except Exception as e:
                if self.is_running:
                    print(f"❌ 音频输入错误: {e}")
                break
                
    async def start_audio_output(self):
        """启动音频输出"""
        # 通过调整播放采样率来改变播放速度
        playback_rate = int(SAMPLE_RATE * PLAYBACK_SPEED)
        
        self.output_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=playback_rate,  # 使用调整后的采样率
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        if PLAYBACK_SPEED != 1.0:
            print(f"🔊 音频输出已启动（{PLAYBACK_SPEED}x 倍速播放）")
        else:
            print("🔊 音频输出已启动")
        
    async def handle_messages(self):
        """处理来自服务器的消息"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                event_type = data.get("type")
                
                # 会话创建
                if event_type == "session.created":
                    self.session_id = data.get("session", {}).get("id")
                    print(f"📝 会话已创建: {self.session_id}")
                
                # 会话更新
                elif event_type == "session.updated":
                    print("✅ 会话配置已更新")
                
                # 输入音频缓冲区已提交
                elif event_type == "input_audio_buffer.committed":
                    print("🎤 用户语音已提交")
                
                # 对话创建
                elif event_type == "conversation.item.created":
                    item = data.get("item", {})
                    if item.get("role") == "user":
                        print("👤 用户消息已创建")
                
                # 响应开始
                elif event_type == "response.created":
                    print("🤖 AI 开始生成回复...")
                    self.cancel_sent = False  # 重置取消标志
                
                # 响应完成
                elif event_type == "response.done":
                    print("✅ AI 回复完成")
                    self.is_ai_speaking = False
                    self.cancel_sent = False
                
                # 音频转录（用户说的话）
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = data.get("transcript", "")
                    print(f"📝 你说: {transcript}")
                
                # AI 文本回复
                elif event_type == "response.text.delta":
                    delta = data.get("delta", "")
                    print(delta, end="", flush=True)
                
                elif event_type == "response.text.done":
                    text = data.get("text", "")
                    if text:
                        print(f"\n💬 AI 回复: {text}")
                
                # 音频数据
                elif event_type == "response.audio.delta":
                    self.is_ai_speaking = True
                    audio_b64 = data.get("delta", "")
                    if audio_b64:
                        # 解码并播放音频
                        audio_data = base64.b64decode(audio_b64)
                        if self.output_stream:
                            self.output_stream.write(audio_data)
                
                elif event_type == "response.audio.done":
                    print("🔊 AI 语音播放完成")
                    self.is_ai_speaking = False
                
                # 响应被取消（打断）
                elif event_type == "response.cancelled":
                    print("⚡ AI 回复已被打断")
                    self.is_ai_speaking = False
                    self.cancel_sent = False
                
                # 错误处理
                elif event_type == "error":
                    error = data.get("error", {})
                    print(f"❌ 错误: {error.get('message', '未知错误')}")
                
        except websockets.exceptions.ConnectionClosed:
            print("🔌 连接已关闭")
        except Exception as e:
            print(f"❌ 消息处理错误: {e}")
            
    async def run(self):
        """运行客户端"""
        try:
            # 连接到服务器
            await self.connect()
            
            # 启动音频输出
            await self.start_audio_output()
            
            # 设置运行标志
            self.is_running = True
            
            print("\n" + "="*60)
            print("🎉 OpenAI Realtime API 客户端已启动")
            print("🎙️  请对着麦克风说话，AI 会实时回复")
            print("🎵 声音: shimmer（温暖友好的女声）")
            print("🌍 多语言模式：中文/英语/日语等自动切换")
            print("⚡ 支持打断：在 AI 说话时，你可以随时打断并说话")
            if ENABLE_ECHO_CANCELLATION:
                print(f"🔇 回音消除已启用（打断阈值: {INTERRUPT_THRESHOLD}）")
            if PLAYBACK_SPEED != 1.0:
                print(f"⚡ 播放速度: {PLAYBACK_SPEED}x")
            if ENABLE_GAIN_CONTROL or ENABLE_NOISE_GATE:
                features = []
                if ENABLE_NOISE_GATE:
                    features.append("降噪")
                if ENABLE_GAIN_CONTROL:
                    features.append("增益")
                print(f"🎛️  音频预处理: {', '.join(features)}")
            print("💡 提示：使用耳机效果最佳，可避免回音")
            print("🛑 按 Ctrl+C 退出")
            print("="*60 + "\n")
            
            # 并发运行音频输入和消息处理
            await asyncio.gather(
                self.start_audio_input(),
                self.handle_messages()
            )
            
        except KeyboardInterrupt:
            print("\n\n👋 正在退出...")
        except Exception as e:
            print(f"❌ 运行错误: {e}")
        finally:
            await self.cleanup()
            
    async def cleanup(self):
        """清理资源"""
        self.is_running = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            
        if self.audio:
            self.audio.terminate()
            
        if self.ws:
            await self.ws.close()
            
        print("✅ 资源已清理")

def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    print("\n\n👋 收到退出信号...")
    sys.exit(0)

async def main():
    """主函数"""
    # 检查 API Key
    if not OPENAI_API_KEY:
        print("❌ 错误: 未找到 OPENAI_API_KEY")
        print("请在 .env 文件中设置你的 API Key")
        return
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    # 创建并运行客户端
    client = RealtimeClient()
    await client.run()

if __name__ == "__main__":
    asyncio.run(main())

