# pdf_processor.py

"""
PDF处理模块
==============
该模块包含PDFConverter类，专门用于将PDF文档转换为Markdown格式。
它使用了 marker-pdf 库来实现高精度的OCR和版面分析。
"""

from pathlib import Path
import marker
from termcolor import colored

class PDFConverter:
    """
    处理PDF到Markdown的转换。
    """
    def __init__(self, pdf_path: Path):
        """
        初始化转换器。

        Args:
            pdf_path (Path): 指向PDF文件的路径。
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件未找到: {pdf_path}")
        self.pdf_path = pdf_path
        self.md_path = pdf_path.with_suffix(".md")

    def to_markdown(self) -> str:
        """
        将PDF转换为Markdown。

        如果一个同名的Markdown文件已经存在，它将使用已缓存的版本以节省时间。

        Returns:
            str: PDF的Markdown内容。
        """
        if self.md_path.exists():
            print(colored(f"♻️ 使用已缓存的Markdown文件: {self.md_path}", "blue"))
            return self.md_path.read_text(encoding='utf-8')

        print(colored(f"🔄 使用Marker将PDF转换为Markdown... (这可能需要一些时间)", "cyan"))
        # marker.convert_single_pdf 返回一个元组 (markdown_text, metadata)
        markdown_text, _ = marker.convert_single_pdf(str(self.pdf_path))

        self.md_path.write_text(markdown_text, encoding='utf-8')
        print(colored(f"✅ PDF成功转换为Markdown: {self.md_path}", "green"))
        return markdown_text