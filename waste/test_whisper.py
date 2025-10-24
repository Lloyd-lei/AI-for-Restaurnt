#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的 Whisper 测试脚本
测试不同模型大小的速度和准确度
"""

import whisper
import pyaudio
import numpy as np
import time
import wave

# 音频参数
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RECORD_SECONDS = 10  # 录音时长

def record_audio(duration=5):
    """录制音频"""
    print(f"\n🎙️  开始录音（{duration}秒，约{duration/60:.1f}分钟）...")
    print("💬 请说话...")
    
    audio = pyaudio.PyAudio()
    
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    frames = []
    total_chunks = int(SAMPLE_RATE / CHUNK_SIZE * duration)
    
    # 显示进度
    import sys
    for i in range(total_chunks):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        frames.append(data)
        
        # 每秒显示一次进度
        if i % (SAMPLE_RATE // CHUNK_SIZE) == 0:
            elapsed = i / (SAMPLE_RATE / CHUNK_SIZE)
            progress = (elapsed / duration) * 100
            sys.stdout.write(f"\r⏱️  录音进度: {elapsed:.0f}/{duration}秒 ({progress:.1f}%)")
            sys.stdout.flush()
    
    print("\n✅ 录音完成！")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # 转换为 numpy 数组
    audio_data = b''.join(frames)
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    return audio_array

def test_model(model_name, audio_data):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"🔍 测试模型: {model_name}")
    print(f"{'='*60}")
    
    # 加载模型
    print("⏳ 正在加载模型...")
    load_start = time.time()
    model = whisper.load_model(model_name)
    load_time = time.time() - load_start
    print(f"✅ 模型加载完成（耗时: {load_time:.2f}秒）")
    
    # 识别
    print("🔍 正在识别...")
    transcribe_start = time.time()
    result = model.transcribe(
        audio_data,
        language=None,  # 自动检测语言
        fp16=False,
        verbose=False
    )
    transcribe_time = time.time() - transcribe_start
    
    # 输出结果
    text = result["text"].strip()
    language = result["language"]
    
    print(f"\n📝 识别结果: {text}")
    print(f"🌍 检测语言: {language}")
    print(f"⏱️  识别耗时: {transcribe_time:.2f}秒")
    print(f"📊 模型大小: {model_name}")
    
    return {
        "model": model_name,
        "text": text,
        "language": language,
        "load_time": load_time,
        "transcribe_time": transcribe_time
    }

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 Whisper 模型")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["tiny", "base"],
        choices=["tiny", "base", "small", "medium", "large", "all"],
        help="要测试的模型（可以指定多个，或使用 'all' 测试所有模型）"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="录音时长（秒），默认5秒。20分钟=1200秒"
    )
    
    args = parser.parse_args()
    
    # 处理 'all' 选项
    if 'all' in args.models:
        models_to_test = ["tiny", "base", "small", "medium", "large"]
    else:
        models_to_test = args.models
    
    print("🎤 Whisper 模型测试工具")
    print(f"📦 将测试模型: {', '.join(models_to_test)}")
    print(f"⏱️  录音时长: {args.duration}秒（约{args.duration/60:.1f}分钟）")
    
    # 估算时间
    total_time_estimate = len(models_to_test) * 5 + args.duration
    print(f"⏰ 预计总耗时: 约{total_time_estimate/60:.1f}分钟")
    
    print("\n💡 模型说明:")
    print("  - tiny: 最快，准确率较低（~39M）")
    print("  - base: 推荐，速度和准确率平衡（~74M）")
    print("  - small: 较准确（~244M）")
    print("  - medium: 很准确（~769M）")
    print("  - large: 最准确，最慢（~1550M）")
    
    if args.duration >= 60:
        print(f"\n⚠️  注意: 录音时长较长（{args.duration/60:.1f}分钟），请确保:")
        print("   1. 麦克风正常工作")
        print("   2. 有足够的存储空间")
        print("   3. 准备好测试内容")
        input("\n按 Enter 键开始录音...")
    
    # 录音
    audio_data = record_audio(args.duration)
    
    # 保存音频文件（可选）
    print("\n💾 正在保存录音文件...")
    import wave
    with wave.open('test_recording.wav', 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        audio_bytes = (audio_data * 32768).astype(np.int16).tobytes()
        wf.writeframes(audio_bytes)
    print("✅ 录音已保存为: test_recording.wav")
    
    # 测试所有模型
    results = []
    for idx, model_name in enumerate(models_to_test, 1):
        print(f"\n{'='*60}")
        print(f"📊 进度: {idx}/{len(models_to_test)}")
        print(f"{'='*60}")
        try:
            result = test_model(model_name, audio_data)
            results.append(result)
        except Exception as e:
            print(f"❌ 模型 {model_name} 测试失败: {e}")
    
    # 汇总结果
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("📊 测试结果对比")
        print(f"{'='*60}")
        
        for r in results:
            print(f"\n🔹 {r['model'].upper()}")
            print(f"   识别: {r['text']}")
            print(f"   语言: {r['language']}")
            print(f"   加载: {r['load_time']:.2f}秒")
            print(f"   识别: {r['transcribe_time']:.2f}秒")
        
        print(f"\n{'='*60}")
        print("💡 建议:")
        print("  - 如果追求速度: 使用 tiny 或 base")
        print("  - 如果追求准确度: 使用 small 或 medium")
        print("  - 实时应用推荐: base（最佳平衡）")

if __name__ == "__main__":
    main()

