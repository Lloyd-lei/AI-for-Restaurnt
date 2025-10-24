#!/bin/bash

echo "🚀 OpenAI Realtime API 客户端安装脚本"
echo "=================================="
echo ""

# 检测操作系统
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📱 检测到 macOS 系统"
    echo "正在检查 PortAudio..."
    
    if ! brew list portaudio &> /dev/null; then
        echo "正在安装 PortAudio..."
        brew install portaudio
    else
        echo "✅ PortAudio 已安装"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 检测到 Linux 系统"
    echo "正在检查 PortAudio..."
    
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y portaudio19-dev python3-pyaudio
    elif command -v yum &> /dev/null; then
        sudo yum install -y portaudio-devel
    fi
fi

echo ""
echo "📦 正在安装 Python 依赖..."
pip3 install -r requirements.txt

echo ""
echo "=================================="
echo "✅ 安装完成！"
echo ""
echo "运行以下命令启动程序："
echo "  python3 realtime_client.py"
echo ""

