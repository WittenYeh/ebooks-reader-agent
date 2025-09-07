# app.py

"""
主应用模块 (Gradio UI)
==============
该文件负责构建和运行Gradio用户界面。
它从其他服务模块（pdf_processor, text_analyzer, video_generator）导入类，
并将UI事件（如按钮点击）与后端功能连接起来。
"""

import gradio as gr
from pathlib import Path
import pandas as pd
import time

# 从解耦后的模块中导入各自的类
from pdf_processor import PDFConverter
from text_analyzer import BookAnalyzer
from video_generator import VideoGenerator

# --- Gradio UI 状态管理 ---
chapter_data_state = gr.State([])
selected_chapter_text_state = gr.State("")

# --- Gradio 配置项 ---
COMFYUI_ADDRESS = "127.0.0.1:8188"
COMFYUI_WORKFLOW_FILE = Path("workflow_video.json")

# --- Gradio 事件处理函数 ---

def process_book_request(
    pdf_file, analysis_type, start_page, end_page, api_key, progress=gr.Progress()
):
    """主处理函数，协调分析流程。"""
    if not api_key: raise gr.Error("需要提供Google API密钥才能继续。")
    if pdf_file is None: raise gr.Error("请先上传一个PDF文件进行分析。")

    work_dir = Path("temp_processing"); work_dir.mkdir(exist_ok=True)
    pdf_path = work_dir / Path(pdf_file.name).name
    pdf_path.write_bytes(pdf_file.read())

    # 定义一个通用的“重置”状态，用于隐藏视频播放器
    hide_video = gr.update(visible=False)

    try:
        if analysis_type == "转换为完整Markdown (使用Marker)":
            converter = PDFConverter(pdf_path=pdf_path)
            markdown_content = converter.to_markdown()
            # 返回6个值，与outputs列表匹配
            return markdown_content, None, None, gr.update(value=[]), "", hide_video

        analyzer = BookAnalyzer(api_key=api_key)
        if analysis_type == "阅读并分析页面范围":
            knowledge_list = analyzer.analyze_page_range(pdf_path, int(start_page), int(end_page))
            if not knowledge_list: return "在指定页面范围内未提取到相关内容。", None, None, gr.update(value=[]), "", hide_video
            output_md = "## 逐页分析结果\n\n"
            for item in knowledge_list:
                output_md += f"### 📄 第 {item['page']} 页\n\n**摘要:** {item['summary']}\n\n**关键点:**\n" + "".join([f"- {p}\n" for p in item['key_points']]) + "\n---\n"
            # 返回6个值，与outputs列表匹配
            return output_md, None, None, gr.update(value=[]), "", hide_video

        elif analysis_type == "通读全书并分割章节":
            book_chapters = analyzer.segment_chapters(pdf_path)
            if not book_chapters or not book_chapters.chapters: return "自动分割章节失败。", None, None, gr.update(value=[]), "", hide_video
            chapters_list = [{"Chapter Title": ch.title, "Page Range": f"{ch.start_page}–{ch.end_page}", "Chapter Summary": ch.summary} for ch in book_chapters.chapters]
            display_df = pd.DataFrame({"Chapter Title": [ch["Chapter Title"] for ch in chapters_list], "Page Range": [ch["Page Range"] for ch in chapters_list]})
            first_summary = chapters_list[0]['Chapter Summary'] if chapters_list else "没有摘要。"
            # *** 修正 ***: 返回6个值，与outputs列表匹配
            return "章节分割完成！请在下方选择一个章节查看详情。", display_df, first_summary, chapters_list, chapters_list[0]['Chapter Summary'], hide_video

    except Exception as e:
        raise gr.Error(f"发生错误: {e}")

def on_select_chapter(evt: gr.SelectData, chapter_data: list):
    """当用户选择一个章节时的回调函数。"""
    if evt.value is None or not chapter_data: return "请选择一个章节查看其摘要。", ""
    selected_summary = chapter_data[evt.index[0]]['Chapter Summary']
    return selected_summary, selected_summary

def generate_video_request(chapter_text, api_key, progress=gr.Progress()):
    """处理生成视频请求的函数。"""
    if not api_key: raise gr.Error("需要提供Google API密钥才能继续。")
    if not chapter_text: raise gr.Error("没有选定的章节内容。请先运行章节分割并选择一个章节。")
    if not COMFYUI_WORKFLOW_FILE.exists(): raise gr.Error(f"ComfyUI工作流文件未找到: {COMFYUI_WORKFLOW_FILE}。")

    progress(0, desc="🚀 初始化视频生成器...")
    try:
        video_generator = VideoGenerator(api_key, COMFYUI_ADDRESS, COMFYUI_WORKFLOW_FILE)
        output_filename = f"chapter_video_{int(time.time())}"
        final_video_path = video_generator.create_chapter_video(chapter_text, output_filename)
        
        if final_video_path:
            return gr.update(value=str(final_video_path), visible=True)
        else:
            raise gr.Error("视频生成失败。请查看终端输出获取更多信息。")
            
    except Exception as e:
        raise gr.Error(f"视频生成过程中发生错误: {e}")

# --- Gradio 界面定义 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 AI 书籍阅读与可视化工具")
    gr.Markdown("上传PDF，将其转换为Markdown，分析页面范围，分割章节，甚至将章节内容可视化为视频！")

    with gr.Row():
        with gr.Column(scale=1):
            api_key_box = gr.Textbox(label="输入你的Google API密钥", type="password", placeholder="以 'AIza...' 开头")
            pdf_upload = gr.File(label="上传PDF文件", file_types=[".pdf"])
            analysis_type_radio = gr.Radio(
                ["阅读并分析页面范围", "通读全书并分割章节", "转换为完整Markdown (使用Marker)"],
                label="选择分析模式", value="阅读并分析页面范围"
            )
            with gr.Group(visible=True) as page_range_group:
                start_page_num = gr.Number(label="起始页", value=1, precision=0)
                end_page_num = gr.Number(label="结束页", value=10, precision=0)
            analysis_type_radio.change(lambda x: gr.update(visible=x == "阅读并分析页面范围"), analysis_type_radio, page_range_group)
            submit_btn = gr.Button("开始分析", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("## 📖 分析与可视化结果")
            output_markdown = gr.Markdown(label="内容概览或完整Markdown")
            
            with gr.Accordion("章节列表 (点击选择)", open=True):
                chapter_df = gr.DataFrame(headers=["章节标题", "页码范围"], datatype=["str", "str"], interactive=True)
                chapter_summary_text = gr.Textbox(label="选定章节的详细摘要", lines=8, interactive=False)
                visualize_btn = gr.Button("生成章节视频 (Visualize Chapter)", variant="secondary")

            output_video = gr.Video(label="生成的章节故事视频", visible=False)

    submit_btn.click(
        fn=process_book_request,
        inputs=[pdf_upload, analysis_type_radio, start_page_num, end_page_num, api_key_box],
        # *** 修正 ***: outputs列表现在包含6个组件，与函数的返回值数量匹配
        outputs=[output_markdown, chapter_df, chapter_summary_text, chapter_data_state, selected_chapter_text_state, output_video]
    )
    chapter_df.select(
        fn=on_select_chapter,
        inputs=[chapter_data_state],
        outputs=[chapter_summary_text, selected_chapter_text_state]
    )
    visualize_btn.click(
        fn=generate_video_request,
        inputs=[selected_chapter_text_state, api_key_box],
        outputs=[output_video]
    )

if __name__ == "__main__":
    demo.launch(debug=True)