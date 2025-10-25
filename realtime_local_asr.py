#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI Realtime 客户端 - 端到端语音对话（支持 Function Calling）
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

# 音频参数（OpenAI Realtime API 要求）
SAMPLE_RATE = 24000  # OpenAI 要求 24kHz
CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16

# 服务器端 VAD 参数（OpenAI 自动检测语音）
VAD_THRESHOLD = 0.5  # 🎯 VAD 敏感度（0.0-1.0），越小越敏感
VAD_PREFIX_PADDING_MS = 300  # 语音前填充（毫秒）
VAD_SILENCE_DURATION_MS = 500  # 🎯 静音多久后结束（毫秒），建议 200-1000

# TTS 参数
TTS_VOICE = "shimmer"  # 🎯 可选: alloy, echo, fable, onyx, nova, shimmer

# ============================================================

class RealtimeClient:
    def __init__(self):
        self.ws = None
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.is_running = False
        self.is_ai_speaking = False
        self.session_id = None
        
        # 打断控制
        self.interrupt_flag = False
        self.drop_audio_until_cancelled = False  # 丢弃音频帧标志
    
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
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {  # 🎯 启用服务器端 VAD（自动检测说话）
                    "type": "server_vad",
                    "threshold": VAD_THRESHOLD,
                    "prefix_padding_ms": VAD_PREFIX_PADDING_MS,
                    "silence_duration_ms": VAD_SILENCE_DURATION_MS
                },
                "tools": FUNCTION_DEFINITIONS  # 🎯 添加 function calling
            }
        }
        
        await self.ws.send(json.dumps(config))
        print(f"⚙️  会话配置已发送（TTS: {TTS_VOICE}，Server VAD 已启用，Functions: {len(FUNCTION_DEFINITIONS)}个）")
        
    async def start_audio_input(self):
        """启动麦克风输入并流式发送到 OpenAI"""
        self.input_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print("🎙️  麦克风已启动，直接流式传输到 OpenAI（Server VAD 自动检测）...")
        
        while self.is_running:
            try:
                # 读取音频
                audio_data = await asyncio.to_thread(
                    self.input_stream.read, CHUNK_SIZE, False
                )
                
                # 🎯 直接发送音频流到 OpenAI（不做本地处理）
                if not self.is_ai_speaking:
                    # Base64 编码
                    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                    
                    # 发送音频帧
                    audio_msg = {
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64
                    }
                    await self.ws.send(json.dumps(audio_msg))
                
                await asyncio.sleep(0.001)
                
            except Exception as e:
                if self.is_running:
                    print(f"❌ 音频输入错误: {e}")
                break
    
    async def _send_function_result(self, call_id, result):
        """发送函数执行结果给 OpenAI"""
        # 重置音频丢弃标志，准备接收新的响应
        self.drop_audio_until_cancelled = False
        
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
        self.output_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,  # 24kHz
            output=True,
            frames_per_buffer=4800
        )
        
        print("🔊 音频输出已启动（24kHz）")
    
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
                        self.output_stream = self.audio.open(
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=SAMPLE_RATE,
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
                
                # 🎯 语音检测事件（Server VAD）
                elif event_type == "input_audio_buffer.speech_started":
                    print("🗣️  检测到语音输入...")
                
                elif event_type == "input_audio_buffer.speech_stopped":
                    print("🔇 语音输入结束，OpenAI 正在识别...")
                
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    # OpenAI 识别完成
                    transcript = data.get("transcript", "")
                    if transcript:
                        print(f"📝 你说: {transcript}")
                
                elif event_type == "input_audio_buffer.committed":
                    print("✅ 音频已提交，等待 AI 回复...")
                
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
                    error_msg = error.get('message', '未知错误')
                    print(f"❌ 错误: {error_msg}")
                    
                    # 如果是取消失败错误，重置音频丢弃标志
                    if "Cancellation failed" in error_msg or "no active response" in error_msg:
                        self.drop_audio_until_cancelled = False
                
        except websockets.exceptions.ConnectionClosed:
            print("🔌 连接已关闭")
        except Exception as e:
            print(f"❌ 消息处理错误: {e}")
            
    async def run(self):
        """运行客户端"""
        try:
            # 连接服务器
            await self.connect()
            
            # 启动音频输出
            await self.start_audio_output()
            
            self.is_running = True
            
            print("\n" + "="*60)
            print("🎉 OpenAI Realtime 客户端已启动（端到端语音对话）")
            print(f"🤖 模型: {MODEL}")
            print(f"🗣️  TTS 语音: {TTS_VOICE}")
            print(f"🎙️  Server VAD 已启用（自动检测语音）")
            print(f"   - VAD 阈值: {VAD_THRESHOLD}")
            print(f"   - 静音检测: {VAD_SILENCE_DURATION_MS}ms")
            print("🌍 多语言自动识别（OpenAI Whisper）")
            print("\n🔧 可用功能:")
            print("  📍 查天气 - 问「北京今天天气怎么样？」")
            print("  🍜 查菜单 - 问「有什么推荐的菜？」「有不辣的菜吗？」")
            print("  📚 搜书籍 - 问「推荐科幻小说」「刘慈欣的书」")
            print("  👋 结束对话 - 说「再见」「拜拜」")
            print("\n💡 操作提示:")
            print("  🎤 直接说话，OpenAI 自动检测和识别（无需等待）")
            print("  ⚡ 按回车键可以打断 AI 说话")
            print("  🛑 按 Ctrl+C 退出")
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
    print("\n\n👋 收到退出信号...")
    sys.exit(0)

async def main():
    """主函数"""
    if not OPENAI_API_KEY:
        print("❌ 错误: 未找到 OPENAI_API_KEY")
        print("请在 .env 文件中设置你的 API Key，或者在代码中直接配置")
        return
    
    print("🚀 OpenAI Realtime 客户端（端到端语音对话）")
    print(f"📦 使用模型: {MODEL}")
    print("")
    
    signal.signal(signal.SIGINT, signal_handler)
    
    client = RealtimeClient()
    await client.run()

if __name__ == "__main__":
    asyncio.run(main())

