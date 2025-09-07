# book_services.py

from pathlib import Path
from typing import List, Optional, Dict, Any

import fitz
import marker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from termcolor import colored

# --- Pydantic Models for Structured LLM Output ---

class PageKnowledge(BaseModel):
    """Data model for knowledge extracted from a single page."""
    has_relevant_content: bool = Field(..., description="True if the page contains story content, dialogue, or plot development; False for tables of contents, blank pages, etc.")
    key_points: List[str] = Field(..., description="A list of key takeaways from the page, focusing on plot, characters, setting, and important dialogue.")
    page_summary: Optional[str] = Field(default=None, description="A brief summary of the core content on this page.")

class Chapter(BaseModel):
    """Data model defining the structure of a single book chapter."""
    title: str = Field(..., description="A concise and representative title for this chapter.")
    summary: str = Field(..., description="A detailed overview of the chapter's content, covering main plot points, character arcs, and key events.")
    start_page: int = Field(..., description="The starting page number of this chapter in the PDF.")
    end_page: int = Field(..., description="The ending page number of this chapter in the PDF.")

class BookChapters(BaseModel):
    """A list containing all chapters of a book."""
    chapters: List[Chapter]


# --- Service Classes ---

class PDFConverter:
    """
    Handles the conversion of a PDF file to a Markdown document using the marker-pdf library.
    """
    def __init__(self, pdf_path: Path):
        """
        Initializes the converter with the path to the PDF file.

        Args:
            pdf_path (Path): The path to the PDF file.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        self.pdf_path = pdf_path
        self.md_path = pdf_path.with_suffix(".md")

    def to_markdown(self) -> str:
        """
        Converts the PDF to Markdown.
        If a Markdown file already exists, it uses the cached version instead of reprocessing.

        Returns:
            str: The Markdown content of the PDF.
        """
        if self.md_path.exists():
            print(colored(f"♻️ Using cached Markdown file: {self.md_path}", "blue"))
            with open(self.md_path, 'r', encoding='utf-8') as f:
                return f.read()

        print(colored(f"🔄 Converting PDF to Markdown with Marker... (This can be slow for large PDFs)", "cyan"))
        # marker.convert_single_pdf returns a tuple (markdown_text, metadata)
        markdown_text, _ = marker.convert_single_pdf(str(self.pdf_path))

        with open(self.md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)

        print(colored(f"✅ PDF successfully converted to Markdown: {self.md_path}", "green"))
        return markdown_text


class BookAnalyzer:
    """
    Handles all interactions with the Large Language Model for book analysis.
    """
    def __init__(self, api_key: str):
        """
        Initializes the analyzer with the user's Google API Key.

        Args:
            api_key (str): The Google API Key for authenticating with the Gemini API.
        """
        self.api_key = api_key
        self.model_name = "gemini-1.5-pro-latest"

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """
        Creates and returns an instance of the ChatGoogleGenerativeAI model.
        """
        return ChatGoogleGenerativeAI(model=self.model_name, temperature=0.2, google_api_key=self.api_key)

    def analyze_page_range(self, pdf_path: Path, start_page: int, end_page: int) -> List[Dict[str, Any]]:
        """
        Analyzes a specific range of pages from a PDF.
        NOTE: This method uses PyMuPDF (fitz) because marker-pdf produces a single
              document, losing the page-specific context required for this feature.

        Args:
            pdf_path (Path): The path to the PDF file.
            start_page (int): The starting page for analysis.
            end_page (int): The ending page for analysis.

        Returns:
            A list of dictionaries, each containing the analysis for a single page.
        """
        llm = self._get_llm()
        pdf_document = fitz.open(pdf_path)
        
        total_pages = pdf_document.page_count
        start_page = max(1, start_page)
        end_page = min(total_pages, end_page)

        if start_page > end_page:
            raise ValueError("Start page cannot be greater than end page.")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a literary analyst. Your task is to read the content of a single book page and extract key information in a structured format. Focus on plot, characters, setting, and important dialogue. Ignore non-story content like indexes or blank pages."),
            ("human", "Analyze the following content from page {page_number}:\n\n---\n\n{page_text}\n\n---")
        ])

        structured_llm = llm.with_structured_output(PageKnowledge)
        chain = prompt | structured_llm
        
        knowledge_base = []
        print(colored(f"\n📖 Analyzing pages {start_page} to {end_page}...", "yellow"))

        for page_num in range(start_page - 1, end_page):
            page_text = pdf_document[page_num].get_text("text")
            if len(page_text.strip()) < 50:
                print(colored(f"⏭️  Skipping page {page_num + 1} (insufficient content)", "yellow"))
                continue

            print(colored(f"🧠 Processing page {page_num + 1}/{total_pages}...", "cyan"))
            try:
                response = chain.invoke({"page_number": page_num + 1, "page_text": page_text})
                if response.has_relevant_content:
                    knowledge_base.append({
                        "page": page_num + 1,
                        "summary": response.page_summary,
                        "key_points": response.key_points,
                    })
                    print(colored(f"✅ Page {page_num + 1} analyzed.", "green"))
            except Exception as e:
                print(colored(f"❌ Error on page {page_num + 1}: {e}", "red"))

        return knowledge_base

    def segment_chapters(self, pdf_path: Path) -> Optional[BookChapters]:
        """
        Analyzes the entire book to identify and segment it into chapters.
        NOTE: This method uses PyMuPDF (fitz) to create a text document with page markers,
              which is crucial for the LLM to determine the start and end pages of each chapter.

        Args:
            pdf_path (Path): The path to the PDF file.

        Returns:
            A BookChapters object containing a list of all identified chapters, or None on failure.
        """
        llm = self._get_llm()
        
        # 1. Generate text with page markers
        pdf_document = fitz.open(pdf_path)
        full_text = "".join([f"\n\n[Page {i + 1}]\n\n{page.get_text('text')}" for i, page in enumerate(pdf_document)])

        # 2. Define prompt and chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional book editor. Your task is to read a book's full text and segment it into logical chapters. For each chapter, provide a title, a detailed summary, and the precise start and end page numbers based on the `[Page X]` markers in the text."),
            ("human", "Here is the full book text:\n\n---\n\n{book_text}\n\n---\n\nPlease segment this into chapters.")
        ])
        
        structured_llm = llm.with_structured_output(BookChapters)
        chain = prompt | structured_llm

        print(colored("\n🤔 Analyzing the full book for chapter segmentation... (This may take several minutes)", "cyan"))
        try:
            response = chain.invoke({"book_text": full_text})
            print(colored("✅ Chapter segmentation successful!", "green"))
            return response
        except Exception as e:
            print(colored(f"❌ Chapter segmentation failed: {e}", "red"))
            return None