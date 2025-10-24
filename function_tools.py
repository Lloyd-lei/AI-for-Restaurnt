#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function Calling 工具函数库
"""

import json
import random
from datetime import datetime

# ============================================================
# 函数实现
# ============================================================

def end_conversation():
    """
    结束对话
    返回: 结束确认消息
    """
    return {
        "status": "conversation_ended",
        "message": "好的，再见！祝你有美好的一天！",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def get_weather(location="北京", date="今天"):
    """
    获取天气信息（模拟数据）
    
    参数:
        location: 城市名称
        date: 日期（今天/明天/后天）
    
    返回: 天气信息
    """
    # 模拟天气数据
    weather_conditions = ["晴", "多云", "阴", "小雨", "大雨", "雪"]
    
    weather_data = {
        "location": location,
        "date": date,
        "condition": random.choice(weather_conditions[:4]),  # 避免极端天气
        "temperature": random.randint(15, 30),
        "temperature_low": random.randint(10, 20),
        "humidity": random.randint(40, 80),
        "wind": random.choice(["微风", "和风", "清风"]),
        "aqi": random.randint(30, 150),
        "suggestion": "适合出行，记得带把伞"
    }
    
    return weather_data

def query_menu(category=None, keyword=None, spicy_level=None, recommend_only=False):
    """
    查询餐厅菜单
    
    参数:
        category: 菜品类别（凉菜/热菜/主食/汤品/饮料）
        keyword: 关键词搜索
        spicy_level: 辣度筛选（不辣/微辣/中辣）
        recommend_only: 只显示推荐菜品
    
    返回: 菜品列表
    """
    try:
        with open('restaurant_menu.json', 'r', encoding='utf-8') as f:
            menu_data = json.load(f)
    except FileNotFoundError:
        return {"error": "菜单文件未找到"}
    
    results = []
    
    # 如果指定了类别
    if category:
        categories = [category] if category in menu_data["菜单"] else menu_data["菜单"].keys()
    else:
        categories = menu_data["菜单"].keys()
    
    # 遍历类别
    for cat in categories:
        for dish in menu_data["菜单"][cat]:
            # 应用筛选条件
            if recommend_only and not dish.get("推荐", False):
                continue
            
            if spicy_level and dish.get("辣度") != spicy_level:
                continue
            
            if keyword and keyword not in dish["名称"] and keyword not in dish.get("描述", ""):
                continue
            
            results.append({
                "类别": cat,
                "名称": dish["名称"],
                "价格": dish["价格"],
                "描述": dish["描述"],
                "辣度": dish.get("辣度", "不辣"),
                "推荐": dish.get("推荐", False)
            })
    
    return {
        "餐厅": menu_data["餐厅名称"],
        "结果数量": len(results),
        "菜品": results[:10]  # 最多返回10个
    }

def search_books(query=None, category=None, author=None, min_rating=None):
    """
    搜索书籍（简单向量搜索模拟）
    
    参数:
        query: 搜索关键词
        category: 书籍类别
        author: 作者名
        min_rating: 最低评分
    
    返回: 书籍列表
    """
    try:
        with open('books_database.json', 'r', encoding='utf-8') as f:
            books_data = json.load(f)
    except FileNotFoundError:
        return {"error": "书籍数据库未找到"}
    
    results = []
    
    for book in books_data["books"]:
        # 应用筛选条件
        if category and book["category"] != category:
            continue
        
        if author and author not in book["author"]:
            continue
        
        if min_rating and book["rating"] < min_rating:
            continue
        
        # 关键词搜索（模拟向量搜索）
        if query:
            query_lower = query.lower()
            # 搜索标题、作者、描述、关键词
            if not (
                query_lower in book["title"].lower() or
                query_lower in book["author"].lower() or
                query_lower in book["description"].lower() or
                any(query_lower in keyword.lower() for keyword in book["keywords"])
            ):
                continue
        
        results.append({
            "书名": book["title"],
            "作者": book["author"],
            "类别": book["category"],
            "年份": book["year"],
            "评分": book["rating"],
            "页数": book["pages"],
            "简介": book["description"],
            "关键词": book["keywords"]
        })
    
    # 按评分排序
    results.sort(key=lambda x: x["评分"], reverse=True)
    
    return {
        "结果数量": len(results),
        "书籍": results[:5]  # 最多返回5本
    }

# ============================================================
# Function Calling 定义（给 OpenAI 的格式）
# ============================================================

FUNCTION_DEFINITIONS = [
    {
        "type": "function",
        "name": "end_conversation",
        "description": "结束对话。当用户明确表示要结束对话、说再见、或者对话已经完成时调用此函数。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "type": "function",
        "name": "get_weather",
        "description": "获取指定城市的天气信息。可以查询今天、明天或后天的天气。",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称，例如：北京、上海、深圳"
                },
                "date": {
                    "type": "string",
                    "description": "日期，可选值：今天、明天、后天",
                    "enum": ["今天", "明天", "后天"]
                }
            },
            "required": ["location"]
        }
    },
    {
        "type": "function",
        "name": "query_menu",
        "description": "查询Meetup中餐厅的菜单。可以按类别、关键词、辣度筛选，或只显示推荐菜品。",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "菜品类别",
                    "enum": ["凉菜", "热菜", "主食", "汤品", "饮料"]
                },
                "keyword": {
                    "type": "string",
                    "description": "搜索关键词，可以是菜名或描述中的词"
                },
                "spicy_level": {
                    "type": "string",
                    "description": "辣度筛选",
                    "enum": ["不辣", "微辣", "中辣"]
                },
                "recommend_only": {
                    "type": "boolean",
                    "description": "是否只显示推荐菜品"
                }
            },
            "required": []
        }
    },
    {
        "type": "function",
        "name": "search_books",
        "description": "搜索书籍数据库。可以按关键词、类别、作者或评分搜索书籍。支持多种类型的书籍：科幻小说、文学小说、历史科普、编程技术等。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，可以是书名、作者、主题或关键词"
                },
                "category": {
                    "type": "string",
                    "description": "书籍类别，如：科幻小说、文学小说、历史科普、编程技术等"
                },
                "author": {
                    "type": "string",
                    "description": "作者名字"
                },
                "min_rating": {
                    "type": "number",
                    "description": "最低评分（0-10）"
                }
            },
            "required": []
        }
    }
]

# 函数映射表
FUNCTION_MAP = {
    "end_conversation": end_conversation,
    "get_weather": get_weather,
    "query_menu": query_menu,
    "search_books": search_books
}

def execute_function(function_name, arguments):
    """
    执行函数调用
    
    参数:
        function_name: 函数名
        arguments: 函数参数（字典）
    
    返回: 函数执行结果
    """
    if function_name not in FUNCTION_MAP:
        return {"error": f"未知函数: {function_name}"}
    
    try:
        func = FUNCTION_MAP[function_name]
        result = func(**arguments)
        return result
    except Exception as e:
        return {"error": f"函数执行错误: {str(e)}"}

