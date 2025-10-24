#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI Realtime 客户端 - 本地 Whisper ASR + OpenAI TTS
"""

import asyncio
import websockets
import json
import pyaudio
import numpy as np
import os
from dotenv import load_dotenv
import signal
import sys
import torch
import whisper
import time
import base64
from function_tools import FUNCTION_DEFINITIONS, execute_function
import threading

# 加载环境变量
load_dotenv()

# ============================================================
# 🎯 配置接口 - 在这里修改设置
# ============================================================

# OpenAI 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_API_URL = "wss://api.openai.com/v1/realtime"
MODEL = "gpt-4o-realtime-preview-2024-10-01"

# 🔧 Whisper 模型选择接口
WHISPER_MODEL = "small"  # 🎯 可选: tiny, base, small, medium, large
# tiny   - 最快，准确率低
# base   - 速度和准确率平衡
# small  - 较准确（推荐）✅
# medium - 很准确但慢
# large  - 最准确但很慢

# 音频参数
SAMPLE_RATE = 16000  # Whisper 要求 16kHz
CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16

# VAD 参数（语音活动检测）
ENERGY_THRESHOLD = 300  # 🎯 语音能量阈值（100-1000），越小越敏感
SILENCE_DURATION = 1.0  # 🎯 静音多久后结束（秒），建议 0.8-2.0
MIN_SPEECH_DURATION = 0.3  # 最小语音时长（秒）

# TTS 参数
TTS_VOICE = "shimmer"  # 🎯 可选: alloy, echo, fable, onyx, nova, shimmer
PLAYBACK_SPEED = 1.0  # 🎯 播放速度（1.0 = 正常，1.2 = 1.2倍速）

# ============================================================

class RealtimeLocalASR:
    def __init__(self):
        self.ws = None
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.is_running = False
        self.is_ai_speaking = False
        self.session_id = None
        
        # Whisper
        self.whisper_model = None
        self.device = "cpu"
        
        # VAD
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_start = None
        
        # 打断控制
        self.interrupt_flag = False
        self.drop_audio_until_cancelled = False  # 丢弃音频帧标志
        
    def load_whisper(self):
        """加载 Whisper 模型"""
        print(f"🔄 正在加载 Whisper 模型: {WHISPER_MODEL}")
        start_time = time.time()
        
        self.whisper_model = whisper.load_model(WHISPER_MODEL, device=self.device)
        
        load_time = time.time() - start_time
        print(f"✅ Whisper 模型加载完成（耗时: {load_time:.2f}秒）")
        
    def calculate_energy(self, audio_data):
        """计算音频能量"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        if len(audio_array) == 0:
            return 0
        energy = np.mean(audio_array.astype(np.float32) ** 2)
        return np.sqrt(energy) if energy > 0 else 0
    
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
        
        await self.configure_session()
        
    async def configure_session(self):
        """配置会话参数"""
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": (
                    "你是一个友好的多语言AI助手，名叫小助手。你可以：\n"
                    "1. 用户说什么语言，你就用什么语言回复\n"
                    "2. 查询天气信息\n"
                    "3. 推荐龙凤楼中餐厅的美食\n"
                    "4. 搜索和推荐书籍\n"
                    "5. 当用户明确表示要结束对话时，调用 end_conversation 函数\n\n"
                    "回复风格：简洁、自然、友好。使用 function calling 来处理具体查询。"
                ),
                "voice": TTS_VOICE,
                "output_audio_format": "pcm16",
                "turn_detection": None,  # 关闭服务器端 VAD，使用本地 VAD
                "tools": FUNCTION_DEFINITIONS  # 🎯 添加 function calling
            }
        }
        
        await self.ws.send(json.dumps(config))
        print(f"⚙️  会话配置已发送（TTS: {TTS_VOICE}，Functions: {len(FUNCTION_DEFINITIONS)}个）")
        
    async def start_audio_input(self):
        """启动麦克风输入"""
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
                # 读取音频
                audio_data = await asyncio.to_thread(
                    self.input_stream.read, CHUNK_SIZE, False
                )
                
                # 只在 AI 不说话时处理用户输入
                if not self.is_ai_speaking:
                    energy = self.calculate_energy(audio_data)
                    
                    # VAD: 检测语音活动
                    if energy > ENERGY_THRESHOLD:
                        if not self.is_speaking:
                            print("🗣️  检测到语音...")
                            self.is_speaking = True
                        
                        self.speech_buffer.append(audio_data)
                        self.silence_start = None
                    else:
                        if self.is_speaking:
                            if self.silence_start is None:
                                self.silence_start = time.time()
                            elif time.time() - self.silence_start > SILENCE_DURATION:
                                # 静音足够长，识别语音
                                print("🔇 语音结束，开始识别...")
                                await self._process_speech()
                                self.speech_buffer = []
                                self.is_speaking = False
                                self.silence_start = None
                        else:
                            # 收集背景音（用于更准确的 VAD）
                            self.speech_buffer.append(audio_data)
                
                await asyncio.sleep(0.001)
                
            except Exception as e:
                if self.is_running:
                    print(f"❌ 音频输入错误: {e}")
                break
                
    async def _process_speech(self):
        """处理并识别语音"""
        if len(self.speech_buffer) < int(SAMPLE_RATE * MIN_SPEECH_DURATION / CHUNK_SIZE):
            print("⚠️  语音太短，跳过")
            return
        
        # 合并音频
        audio_data = b''.join(self.speech_buffer)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        print("🔍 正在识别...")
        recognize_start = time.time()
        
        try:
            # Whisper 识别
            result = await asyncio.to_thread(
                self.whisper_model.transcribe,
                audio_array,
                language=None,  # 自动检测语言
                fp16=False,
                verbose=False
            )
            
            recognize_time = time.time() - recognize_start
            text = result["text"].strip()
            language = result["language"]
            
            if text:
                print(f"📝 你说: {text} [{language}] (耗时: {recognize_time:.2f}秒)")
                # 发送文本给 OpenAI
                await self._send_text_to_openai(text)
            else:
                print("⚠️  未识别到内容")
            
        except Exception as e:
            print(f"❌ 识别错误: {e}")
    
    async def _send_text_to_openai(self, text):
        """发送文本给 OpenAI 并请求音频回复"""
        # 创建对话项
        message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text
                    }
                ]
            }
        }
        await self.ws.send(json.dumps(message))
        
        # 请求响应
        response_msg = {
            "type": "response.create",
            "response": {
                "modalities": ["audio", "text"]
            }
        }
        await self.ws.send(json.dumps(response_msg))
    
    async def _send_function_result(self, call_id, result):
        """发送函数执行结果给 OpenAI"""
        message = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result, ensure_ascii=False)
            }
        }
        await self.ws.send(json.dumps(message))
        
        # 请求 AI 继续响应
        response_msg = {
            "type": "response.create"
        }
        await self.ws.send(json.dumps(response_msg))
        
    async def start_audio_output(self):
        """启动音频输出"""
        playback_rate = int(24000 * PLAYBACK_SPEED)  # OpenAI 输出是 24kHz
        
        self.output_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=playback_rate,
            output=True,
            frames_per_buffer=4800
        )
        
        if PLAYBACK_SPEED != 1.0:
            print(f"🔊 音频输出已启动（{PLAYBACK_SPEED}x 倍速）")
        else:
            print("🔊 音频输出已启动")
    
    async def keyboard_listener(self):
        """监听键盘输入（按回车打断）"""
        def listen_keyboard():
            while self.is_running:
                try:
                    input()  # 等待回车键
                    if self.is_ai_speaking:
                        self.interrupt_flag = True
                except:
                    break
        
        # 在单独的线程中运行
        thread = threading.Thread(target=listen_keyboard, daemon=True)
        thread.start()
        
        # 检查打断标志
        while self.is_running:
            if self.interrupt_flag:
                print("\n⚡ 检测到打断（回车键），立即停止播放")
                
                # 🎯 1. 立即设置标志，丢弃后续音频帧（不等服务器确认）
                self.drop_audio_until_cancelled = True
                self.is_ai_speaking = False
                
                # 🎯 2. 立即清空音频缓冲区并重启流
                if self.output_stream:
                    try:
                        self.output_stream.stop_stream()
                        self.output_stream.close()
                        
                        # 重新创建音频流（彻底清空缓冲区）
                        playback_rate = int(24000 * PLAYBACK_SPEED)
                        self.output_stream = self.audio.open(
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=playback_rate,
                            output=True,
                            frames_per_buffer=4800
                        )
                        print("🔇 音频缓冲区已清空")
                    except Exception as e:
                        print(f"⚠️  重置音频流: {e}")
                
                # 🎯 3. 异步发送取消消息到服务器（不阻塞）
                try:
                    cancel_msg = {
                        "type": "response.cancel"
                    }
                    await self.ws.send(json.dumps(cancel_msg))
                    print("📤 已发送取消请求到服务器（等待确认）")
                except Exception as e:
                    print(f"❌ 发送取消请求失败: {e}")
                    # 即使发送失败，本地也已经停止了
                    self.drop_audio_until_cancelled = False
                
                self.interrupt_flag = False
                print("🎙️  已恢复监听，可以继续说话...")
            
            await asyncio.sleep(0.05)  # 更快的检查频率（50ms）
        
    async def handle_messages(self):
        """处理来自服务器的消息"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                event_type = data.get("type")
                
                if event_type == "session.created":
                    self.session_id = data.get("session", {}).get("id")
                    print(f"📝 会话已创建: {self.session_id}")
                
                elif event_type == "session.updated":
                    print("✅ 会话配置已更新")
                
                elif event_type == "response.created":
                    print("🤖 AI 开始生成回复...")
                
                elif event_type == "response.done":
                    print("✅ AI 回复完成\n")
                    self.is_ai_speaking = False
                
                elif event_type == "response.text.delta":
                    delta = data.get("delta", "")
                    print(delta, end="", flush=True)
                
                elif event_type == "response.text.done":
                    text = data.get("text", "")
                    if text:
                        print(f"\n💬 AI: {text}")
                
                elif event_type == "response.audio.delta":
                    # 🎯 如果正在等待取消确认，丢弃所有音频帧
                    if self.drop_audio_until_cancelled:
                        # 静默丢弃，不播放
                        continue
                    
                    self.is_ai_speaking = True
                    audio_b64 = data.get("delta", "")
                    if audio_b64 and self.output_stream:
                        audio_data = base64.b64decode(audio_b64)
                        try:
                            self.output_stream.write(audio_data)
                        except Exception as e:
                            # 可能在重置流时出错，忽略
                            pass
                
                elif event_type == "response.audio.done":
                    print("🔊 AI 语音播放完成")
                    self.is_ai_speaking = False
                
                # 🎯 Function call 相关事件
                elif event_type == "response.function_call_arguments.delta":
                    # Function 参数增量（可选：显示参数构建过程）
                    pass
                
                elif event_type == "response.function_call_arguments.done":
                    # Function 参数接收完成
                    call_id = data.get("call_id")
                    function_name = data.get("name")
                    arguments_str = data.get("arguments", "{}")
                    
                    print(f"\n🔧 调用函数: {function_name}")
                    print(f"📋 参数: {arguments_str}")
                    
                    # 执行函数
                    try:
                        arguments = json.loads(arguments_str)
                        result = execute_function(function_name, arguments)
                        
                        print(f"✅ 函数结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
                        
                        # 🎯 特殊处理：end_conversation
                        if function_name == "end_conversation":
                            print("\n👋 用户选择结束对话")
                            # 先发送函数结果，然后退出
                            await self._send_function_result(call_id, result)
                            await asyncio.sleep(2)  # 等待 AI 说完再见
                            self.is_running = False
                            return
                        
                        # 发送函数结果给 OpenAI
                        await self._send_function_result(call_id, result)
                        
                    except Exception as e:
                        print(f"❌ 函数执行失败: {e}")
                        error_result = {"error": str(e)}
                        await self._send_function_result(call_id, error_result)
                
                # 🎯 响应被取消（打断）
                elif event_type == "response.cancelled":
                    print("✅ 服务器确认：响应已取消")
                    self.is_ai_speaking = False
                    self.drop_audio_until_cancelled = False  # 重置丢弃标志
                
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
            # 加载 Whisper
            self.load_whisper()
            
            # 连接服务器
            await self.connect()
            
            # 启动音频输出
            await self.start_audio_output()
            
            self.is_running = True
            
            print("\n" + "="*60)
            print("🎉 Realtime 客户端已启动（本地 ASR + Function Calling）")
            print(f"🎙️  本地 ASR: Whisper {WHISPER_MODEL.upper()}")
            print(f"🗣️  OpenAI TTS: {TTS_VOICE}")
            print(f"🎚️  VAD 阈值: {ENERGY_THRESHOLD}")
            print(f"⏱️  静音时长: {SILENCE_DURATION}秒")
            print("🌍 多语言自动识别")
            print("\n🔧 可用功能:")
            print("  📍 查天气 - 问「北京今天天气怎么样？」")
            print("  🍜 查菜单 - 问「有什么推荐的菜？」「有不辣的菜吗？」")
            print("  📚 搜书籍 - 问「推荐科幻小说」「刘慈欣的书」")
            print("  👋 结束对话 - 说「再见」「拜拜」")
            print("\n💡 操作提示:")
            print("  🎤 说话后停顿即可自动识别")
            print("  ⚡ 按回车键可以取消 AI 响应（停止 LLM+TTS 生成）")
            print("  🛑 按 Ctrl+C 强制退出")
            print("="*60 + "\n")
            
            # 运行（添加键盘监听）
            await asyncio.gather(
                self.start_audio_input(),
                self.handle_messages(),
                self.keyboard_listener()
            )
            
        except KeyboardInterrupt:
            print("\n\n👋 正在退出...")
        except Exception as e:
            print(f"❌ 运行错误: {e}")
            import traceback
            traceback.print_exc()
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
