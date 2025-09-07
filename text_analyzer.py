# text_analyzer.py

"""
文本分析模块 (已移除 LangChain)
================================
该模块包含BookAnalyzer类，负责与谷歌Gemini API直接交互，
以执行文本分析任务，包括：
- 按页面范围分析内容
- 将整本书分割成逻辑章节
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import fitz
import google.generativeai as genai # 移除了 langchain, 导入官方库
from pydantic import BaseModel, Field
from termcolor import colored

# --- Pydantic 模型定义 (保持不变) ---

class PageKnowledge(BaseModel):
    """用于从单个页面提取知识的模型"""
    has_relevant_content: bool = Field(..., description="如果页面包含故事内容、对话或情节发展，则为True；否则为False（例如目录、空白页）")
    key_points: List[str] = Field(..., description="从页面提取的关键知识点列表，重点关注情节、角色、设定和重要对话。")
    page_summary: Optional[str] = Field(default=None, description="对该页核心内容的简短概括。")

class Chapter(BaseModel):
    """定义一本书的单个章节结构"""
    title: str = Field(..., description="为本章节生成一个简洁且有代表性的标题。")
    summary: str = Field(..., description="对本章节所有内容的详细概览，包括主要情节、角色发展和关键事件。")
    start_page: int = Field(..., description="该章节在PDF中的起始页码。")
    end_page: int = Field(..., description="该章节在PDF中的结束页码。")

class BookChapters(BaseModel):
    """包含一本书所有章节的列表"""
    chapters: List[Chapter]

# --- 服务类 ---

class BookAnalyzer:
    """处理所有与LLM相关的书籍文本分析任务 (使用 google-generativeai)"""
    def __init__(self, api_key: str):
        # 配置官方库的API密钥
        genai.configure(api_key=api_key)
        self.model_name = "gemini-1.5-pro-latest"
        # 配置模型以强制输出JSON
        self.generation_config = genai.GenerationConfig(response_mime_type="application/json")

    def _create_prompt(self, system_instruction: str, user_content: str, schema: Dict) -> str:
        """创建一个包含JSON schema的完整提示词"""
        return f"{system_instruction}\n\n请严格按照下面的JSON Schema格式返回你的分析结果:\n{json.dumps(schema, indent=2)}\n\n需要分析的文本如下:\n---\n{user_content}"

    def analyze_page_range(self, pdf_path: Path, start_page: int, end_page: int) -> List[Dict[str, Any]]:
        """分析PDF中一个指定的页面范围。"""
        model = genai.GenerativeModel(self.model_name, generation_config=self.generation_config)
        pdf_document = fitz.open(pdf_path)
        total_pages = pdf_document.page_count
        start_page = max(1, start_page)
        end_page = min(total_pages, end_page)

        if start_page > end_page:
            raise ValueError("起始页不能大于结束页。")

        system_instruction = "你是一位文学分析师。你的任务是阅读单页书本内容，并以结构化JSON格式提取关键信息。请关注情节、人物、背景和重要对话。忽略目录或空白页等非故事内容。"
        
        knowledge_base = []
        print(colored(f"\n📖 正在分析第 {start_page} 页到 {end_page} 页...", "yellow"))

        for page_num in range(start_page - 1, end_page):
            page_text = pdf_document[page_num].get_text("text")
            if len(page_text.strip()) < 50:
                continue
            
            print(colored(f"🧠 正在处理第 {page_num + 1}/{total_pages} 页...", "cyan"))
            try:
                # 创建包含Schema的提示词
                prompt = self._create_prompt(
                    system_instruction,
                    f"这是来自第 {page_num + 1} 页的内容:\n{page_text}",
                    PageKnowledge.model_json_schema()
                )
                # 调用API
                response = model.generate_content(prompt)
                # 使用Pydantic解析和验证JSON
                page_knowledge = PageKnowledge.model_validate_json(response.text)
                
                if page_knowledge.has_relevant_content:
                    knowledge_base.append({"page": page_num + 1, "summary": page_knowledge.page_summary, "key_points": page_knowledge.key_points})
                    print(colored(f"✅ 第 {page_num + 1} 页分析完毕。", "green"))
            except Exception as e:
                print(colored(f"❌ 第 {page_num + 1} 页出错: {e}", "red"))

        return knowledge_base

    def segment_chapters(self, pdf_path: Path) -> Optional[BookChapters]:
        """分析整本书以识别和分割章节。"""
        model = genai.GenerativeModel(self.model_name, generation_config=self.generation_config)
        pdf_document = fitz.open(pdf_path)
        full_text = "".join([f"\n\n[Page {i + 1}]\n\n{page.get_text('text')}" for i, page in enumerate(pdf_document)])
        
        system_instruction = "你是一位专业的图书编辑。你的任务是阅读一本书的全文，并将其分割成逻辑清晰的章节。对于每个章节，请根据文本中的 `[Page X]` 标记，提供标题、详细摘要以及精确的起止页码。"
        
        print(colored("\n🤔 正在分析全书以分割章节... (这可能需要几分钟)", "cyan"))
        try:
            prompt = self._create_prompt(
                system_instruction,
                full_text,
                BookChapters.model_json_schema()
            )
            response = model.generate_content(prompt)
            # 解析和验证JSON
            book_chapters = BookChapters.model_validate_json(response.text)
            
            print(colored("✅ 章节分割成功!", "green"))
            return book_chapters
        except Exception as e:
            print(colored(f"❌ 章节分割失败: {e}", "red"))
            return None