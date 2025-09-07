# video_generator.py

"""
视频生成模块 (已移除 LangChain)
================================
该模块包含VideoGenerator类，负责将文本章节转换为可视化视频。
其工作流程包括：
1. 使用Gemini API将章节文本分解为一系列可视化场景。
2. 通过WebSocket API与ComfyUI服务通信，为每个场景生成视频片段。
3. 使用moviepy将所有视频片段拼接成一个完整的视频。
"""

import json
import time
import uuid
import requests
import websocket
from pathlib import Path
from typing import List, Optional, Dict

import google.generativeai as genai
from pydantic import BaseModel, Field
from moviepy.editor import VideoFileClip, concatenate_videoclips
from termcolor import colored

# --- Pydantic 模型定义 ---

class Scene(BaseModel):
    """定义视频故事板的单个场景"""
    scene_description: str = Field(..., description="对这个场景的简短文本描述，概括发生了什么。")
    visual_prompt: str = Field(..., description="一个为AI视频生成模型优化的、详细的视觉提示词。应包含主体、动作、环境、风格等。")

class ChapterScenes(BaseModel):
    """包含一个章节所有场景的列表"""
    scenes: List[Scene]

# --- 服务类 ---

class VideoGenerator:
    """处理所有与视频生成相关的任务 (使用 google-generativeai)"""
    def __init__(self, api_key: str, comfyui_address: str, workflow_path: Path):
        genai.configure(api_key=api_key)
        self.model_name = "gemini-1.5-pro-latest"
        self.generation_config = genai.GenerationConfig(response_mime_type="application/json")
        
        self.server_address = comfyui_address
        self.client_id = str(uuid.uuid4())
        
        if not workflow_path.exists():
            raise FileNotFoundError(f"ComfyUI工作流文件未找到: {workflow_path}")
        self.base_workflow = json.loads(workflow_path.read_text(encoding='utf-8'))

    def _create_prompt(self, system_instruction: str, user_content: str, schema: Dict) -> str:
        """创建一个包含JSON schema的完整提示词"""
        return f"{system_instruction}\n\n请严格按照下面的JSON Schema格式返回你的分析结果:\n{json.dumps(schema, indent=2)}\n\n需要分析的文本如下:\n---\n{user_content}"

    def split_chapter_into_scenes(self, chapter_text: str) -> Optional[ChapterScenes]:
        """使用LLM将章节文本分割成一系列可视化场景"""
        model = genai.GenerativeModel(self.model_name, generation_config=self.generation_config)
        system_instruction = """你是一位电影导演和故事板画师。你的任务是将下面的章节文本分解成一系列独立的、可视化的场景。
            对于每个场景，完成两件事：
            1.  `scene_description`: 用一句话简要描述这个场景的核心内容。
            2.  `visual_prompt`: 创作一个详细、生动的提示词，供AI视频生成模型使用。这个提示词应该包含场景、角色、动作、情绪和艺术风格。例如："cinematic shot, Alice falling down a rabbit hole, swirling vortex of colors and objects, surreal, dreamlike, high detail"。
            确保场景数量在5到10个之间，以保持视频节奏。"""

        print(colored("\n🎬 正在将章节分割为可视化场景...", "cyan"))
        try:
            prompt = self._create_prompt(system_instruction, chapter_text, ChapterScenes.model_json_schema())
            response = model.generate_content(prompt)
            scenes = ChapterScenes.model_validate_json(response.text)
            
            print(colored(f"✅ 成功将章节分割为 {len(scenes.scenes)} 个场景。", "green"))
            return scenes
        except Exception as e:
            print(colored(f"❌ 场景分割失败: {e}", "red"))
            return None
    
    def _queue_comfyui_prompt(self, prompt_workflow: Dict) -> Dict:
        """将一个任务添加到ComfyUI队列中，并等待结果"""
        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
        ws.send(json.dumps({"prompt": prompt_workflow, "client_id": self.client_id}))
        
        print(colored("⏳ 等待ComfyUI生成视频片段...", "yellow"))
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executed':
                    ws.close()
                    return message['data']['output']
    
    def _generate_clip_for_scene(self, scene_prompt: str, prompt_node_title: str) -> Optional[Path]:
        """使用ComfyUI为单个场景生成视频片段"""
        workflow = json.loads(json.dumps(self.base_workflow))
        target_node_id = next((nid for nid, n in workflow.items() if n.get("_meta", {}).get("title") == prompt_node_title), None)
        
        if not target_node_id:
            raise ValueError(f"在工作流中找不到标题为 '{prompt_node_title}' 的节点。")
        
        workflow[target_node_id]["inputs"]["text"] = scene_prompt
        outputs = self._queue_comfyui_prompt(workflow)
        
        for node_id in outputs:
            if 'videos' in outputs[node_id]:
                video_data = outputs[node_id]['videos'][0]
                url = f"http://{self.server_address}/view?subfolder={video_data.get('subfolder', '')}&filename={video_data['filename']}"
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    output_dir = Path("temp_clips")
                    output_dir.mkdir(exist_ok=True)
                    clip_path = output_dir / video_data['filename']
                    clip_path.write_bytes(response.content)
                    print(colored(f"\n✅ 视频片段已保存: {clip_path}", "green"))
                    return clip_path
        return None

    def _stitch_clips_into_video(self, clip_paths: List[Path], output_filename: str) -> Path:
        """将视频片段拼接成一个完整的视频"""
        print(colored("\n🎞️ 正在拼接所有视频片段...", "cyan"))
        clips = [VideoFileClip(str(p)) for p in clip_paths]
        final_clip = concatenate_videoclips(clips, method="compose")
        
        output_dir = Path("final_videos")
        output_dir.mkdir(exist_ok=True)
        final_path = output_dir / f"{output_filename}.mp4"
        
        # *** 修正 ***: 将错误的 'libx24' 修正为正确的 'libx264'
        final_clip.write_videofile(str(final_path), codec="libx264", audio_codec="aac")
        print(colored(f"✅ 最终视频已生成: {final_path}", "green"))

        for p in clip_paths: p.unlink()
        return final_path

    def create_chapter_video(self, chapter_text: str, output_filename: str = "chapter_video") -> Optional[Path]:
        """创建章节视频的完整流程。"""
        scenes_data = self.split_chapter_into_scenes(chapter_text)
        if not scenes_data or not scenes_data.scenes: return None
        
        clip_paths = []
        for i, scene in enumerate(scenes_data.scenes):
            print(colored(f"\n🎬 正在生成场景 {i+1}/{len(scenes_data.scenes)}: '{scene.scene_description}'", "magenta"))
            clip_path = self._generate_clip_for_scene(scene.visual_prompt, "Prompt_Input_Node")
            if clip_path:
                clip_paths.append(clip_path)

        if not clip_paths:
            print(colored("❌ 未能生成任何视频片段，无法创建最终视频。", "red"))
            return None
            
        return self._stitch_clips_into_video(clip_paths, output_filename)