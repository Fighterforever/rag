#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本分块器模块
负责将结构化文档元素智能分块，保留文档结构和语义完整性
"""

import os
import logging
import re
from typing import Dict, List, Optional, Tuple, Union, Any
import json
from pathlib import Path
import PIL.Image
# 导入Google Gemini API
from google import genai

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextChunker:
    """
    智能文本分块器，基于文档结构和语义边界进行分块
    """
    
    def __init__(
        self,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 100,
        chunk_overlap: int = 200,
        preserve_tables: bool = True,
        preserve_images: bool = True,
        respect_headers: bool = True,
        gemini_api_key: Optional[str] = None,
        filter_headers: bool = True,  # 新增：是否过滤无关标题
        main_content_only: bool = True  # 新增：是否只保留主要正文内容
    ):
        """
        初始化文本分块器
        
        Args:
            max_chunk_size: 块的最大字符数
            min_chunk_size: 块的最小字符数
            chunk_overlap: 块之间的重叠字符数
            preserve_tables: 是否保留表格完整性
            preserve_images: 是否保留图像完整性
            respect_headers: 是否尊重标题作为分块边界
            gemini_api_key: Google Gemini API密钥，用于图片摘要
            filter_headers: 是否过滤掉无关的标题或非正文部分
            main_content_only: 是否只保留文档主要正文
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_tables = preserve_tables
        self.preserve_images = preserve_images
        self.respect_headers = respect_headers
        self.gemini_api_key = gemini_api_key
        self.gemini_client = None
        self.filter_headers = filter_headers
        self.main_content_only = main_content_only
        
        # 标题过滤模式：针对常见的需要过滤的标题
        self.irrelevant_title_patterns = [
            r'^目录$', r'^索引$', r'^附录', r'^参考文献$', r'^引言$',
            r'^前言$', r'^声明$', r'^致谢$', r'^摘要$', r'^abstract$',
            r'^版权', r'^版本历史', r'^修订记录', r'^文档信息$',
            r'^页眉$', r'^页脚$', r'^链接$', r'^references$'
        ]
        
        # 初始化Gemini客户端（如果提供了API密钥）
        if self.gemini_api_key:
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
                logger.info("Google Gemini API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Google Gemini API client: {str(e)}")
                self.gemini_client = None
        
        # 定义用于匹配Markdown图片链接的正则表达式
        self.img_pattern = re.compile(r'(?i)<img\s+(?:[^>]*?\s+)?src\s*=\s*(?:(["\'])(.*?)\1|([^">\s]+))[^>]*?>')
        
        logger.info(
            "TextChunker initialized with max_size=%d, min_size=%d, overlap=%d, "
            "preserve_tables=%s, preserve_images=%s, respect_headers=%s, gemini_api=%s, "
            "filter_headers=%s, main_content_only=%s",
            max_chunk_size, min_chunk_size, chunk_overlap,
            preserve_tables, preserve_images, respect_headers,
            "enabled" if self.gemini_client else "disabled",
            filter_headers, main_content_only
        )
    
    def create_chunks(self, elements: List[Dict[str, Any]], original_file_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        将文档元素分块，保留结构和语义完整性
        
        Args:
            elements: 文档元素列表，每个元素为包含类型、内容和元数据的字典
            original_file_path: 原始文件路径，用于解析图片的相对路径
            
        Returns:
            分块后的文本块列表，每个块为一个字典，包含内容和元数据
        """
        if not elements:
            logger.warning("Empty elements list provided")
            return []
        
        # 存储原始文件路径，用于图片路径解析
        self.original_file_path = original_file_path
        
        # 预处理元素内容，处理图片链接
        if self.gemini_client and original_file_path:
            self._preprocess_elements_with_image_summaries(elements)
        
        # 如果启用了过滤，对元素进行过滤处理
        if self.filter_headers or self.main_content_only:
            elements = self._filter_elements(elements)
            if not elements:
                logger.warning("No elements remained after filtering")
                return []
        
        # 首先按结构对元素进行预处理和分组
        element_groups = self._preprocess_and_group_elements(elements)
        
        # 创建最终的文本块
        chunks = []
        for group in element_groups:
            group_chunks = self._chunk_element_group(group)
            chunks.extend(group_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(elements)} elements")
        return chunks
    
    def _filter_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        过滤元素列表，移除无关的标题或非主要内容
        
        Args:
            elements: 文档元素列表
            
        Returns:
            过滤后的元素列表
        """
        # 如果不启用过滤，直接返回原始元素
        if not self.filter_headers and not self.main_content_only:
            return elements
            
        filtered_elements = []
        main_content_started = False
        main_content_ended = False
        
        # 找出可能的主要内容区域（对于某些特定格式的文档）
        if self.main_content_only:
            # 尝试识别主要内容区域的开始和结束
            for i, element in enumerate(elements):
                element_type = element.get("type", "")
                content = element.get("content", "").strip()
                
                # 检测可能的主要内容开始标志
                if (not main_content_started and 
                    element_type == "Title" and 
                    (re.search(r'^第.*章|^\d+[\.、]|^引言$|^正文|^概述|^简介|^背景$', content) or 
                     len(content) < 20 and i > 5)):  # 允许一些通用的章节标题开始主要内容
                    main_content_started = True
                
                # 检测可能的主要内容结束标志
                if (main_content_started and not main_content_ended and 
                    element_type == "Title" and 
                    re.search(r'^参考|^附录|^附件|^索引$|^结语$|^结论$|^致谢$', content)):
                    main_content_ended = True
        else:
            # 如果不筛选主要内容，则认为整个文档都是主要内容
            main_content_started = True
        
        # 过滤并处理元素
        for element in elements:
            element_type = element.get("type", "")
            content = element.get("content", "").strip()
            
            # 处理标题元素
            if element_type == "Title" and self.filter_headers:
                # 检查是否为无关标题
                if any(re.search(pattern, content, re.IGNORECASE) for pattern in self.irrelevant_title_patterns):
                    logger.debug(f"Filtered out irrelevant title: {content}")
                    continue
            
            # 如果只保留主要内容，需要确认当前元素是否在主要内容区域内
            if self.main_content_only:
                if not main_content_started:
                    continue
                if main_content_ended:
                    # 例外：表格和图像在主要内容之后也可能包含重要信息
                    metadata = element.get("metadata", {})
                    if not (metadata.get("is_table", False) or metadata.get("is_image", False)):
                        continue
            
            # 将通过过滤的元素添加到新列表
            filtered_elements.append(element)
        
        logger.info(f"Filtered elements: {len(elements)} -> {len(filtered_elements)}")
        return filtered_elements
    
    def _preprocess_elements_with_image_summaries(self, elements: List[Dict[str, Any]]) -> None:
        """
        预处理元素内容，将图片链接替换为图片摘要
        
        Args:
            elements: 文档元素列表
        """
        for element in elements:
            if element.get("type") == "Text" or element.get("type") == "Paragraph":
                content = element.get("content", "")
                if self.img_pattern.search(content):
                    # 处理内容中的图片链接
                    new_content = self._process_element_content(content)
                    element["content"] = new_content
    
    def _process_element_content(self, content: str) -> str:
        """
        处理元素内容，将图片链接替换为图片摘要
        
        Args:
            content: 元素内容
            
        Returns:
            处理后的内容
        """
        if not self.gemini_client or not self.original_file_path:
            return content
        
        # 找到所有图片链接
        matches = list(self.img_pattern.finditer(content))
        if not matches:
            return content
        
        # 从后向前替换，以保持索引的有效性
        for match in reversed(matches):
            img_path = match.group(1)
            img_tag = match.group(0)
            start_idx, end_idx = match.span()
            
            # 获取图片摘要
            summary = self._get_gemini_image_summary(img_path)
            
            if summary:
                # 替换为图片摘要
                content = content[:start_idx] + f"[图片摘要: {summary}]" + content[end_idx:]
            else:
                # 如果无法获取摘要，保留原图片链接，但添加注释
                content = content[:start_idx] + f"[未处理的图片: {img_path}]" + content[end_idx:]
        
        return content
    
    def _get_gemini_image_summary(self, img_path: str) -> Optional[str]:
        """
        调用Google Gemini API获取图片摘要
        
        Args:
            img_path: 图片路径（可能是相对路径）
            
        Returns:
            图片摘要，如果失败则返回None
        """
        if not self.gemini_client or not self.original_file_path:
            return None
        
        try:
            # 构建图片的完整路径
            # 处理不同的路径格式，支持绝对路径和相对路径
            if os.path.isabs(img_path):
                full_img_path = Path(img_path)
            else:
                # 如果是相对路径，尝试多种可能的基础路径
                parent_dir = self.original_file_path.parent
                
                # 可能的图片路径组合：
                # 1. 与Markdown文件位于同一目录
                path1 = parent_dir / img_path
                
                # 2. 与Markdown文件的文件名同名的子目录中
                md_name = self.original_file_path.stem
                path2 = parent_dir / md_name / Path(img_path).name
                
                # 3. 已经包含了子目录名称
                path3 = parent_dir / img_path.lstrip('/')
                
                # 检查哪个路径存在
                if path1.exists():
                    full_img_path = path1
                elif path2.exists():
                    full_img_path = path2
                elif path3.exists():
                    full_img_path = path3
                else:
                    # 如果都不存在，尝试一种特殊情况：img_path的文件名部分作为相对路径
                    special_case = parent_dir / Path(img_path).name
                    if special_case.exists():
                        full_img_path = special_case
                    else:
                        logger.warning(f"Image not found: tried {path1}, {path2}, {path3}, and {special_case}")
                        return None
            
            # 检查文件是否存在
            if not full_img_path.exists():
                logger.warning(f"Image file not found: {full_img_path}")
                return None
            
            logger.info(f"Processing image: {full_img_path}")
            
            # 加载图片
            image = PIL.Image.open(full_img_path)
            
            # 调用Gemini API获取图片摘要
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-pro-exp-03-25",
                contents=["请用简洁的中文对这张图片进行摘要描述", image]
            )
            
            # 获取摘要文本
            summary = response.text.strip()
            logger.info(f"Generated image summary for {img_path}: {summary[:50]}...")
            
            return summary
        
        except Exception as e:
            logger.error(f"Error getting image summary for {img_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _preprocess_and_group_elements(self, elements: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        预处理元素并按结构分组
        
        Args:
            elements: 文档元素列表
            
        Returns:
            按结构分组的元素组列表
        """
        # 对元素按类型和标题路径进行分组
        groups = []
        current_group = []
        current_header_path = []
        
        for element in elements:
            element_type = element.get("type", "")
            
            # 获取元素的标题路径
            header_path = element.get("metadata", {}).get("heading_path", [])
            
            # 特殊处理表格和图像
            is_table = element.get("metadata", {}).get("is_table", False)
            is_image = element.get("metadata", {}).get("is_image", False)
            
            # 检查是否应该开始新的分组
            start_new_group = False
            
            # 当遇到标题或标题路径变化时开始新的分组
            if self.respect_headers and element_type == "Title":
                start_new_group = True
            # 标题路径变化时也开始新的分组
            elif self.respect_headers and header_path != current_header_path:
                start_new_group = True
            # 表格和图像作为独立分组处理
            elif (is_table and self.preserve_tables) or (is_image and self.preserve_images):
                # 如果当前组不为空，先保存当前组
                if current_group:
                    groups.append(current_group)
                    current_group = []
                # 创建独立分组后添加
                groups.append([element])
                continue
            
            if start_new_group and current_group:
                groups.append(current_group)
                current_group = []
            
            current_group.append(element)
            current_header_path = header_path
        
        # 添加最后一个分组
        if current_group:
            groups.append(current_group)
        
        logger.debug(f"Created {len(groups)} element groups")
        return groups
    
    def _chunk_element_group(self, group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将一个元素组分块
        
        Args:
            group: 一个元素组
            
        Returns:
            分块后的文本块列表
        """
        if not group:
            return []
        
        # 如果组中只有一个元素且是表格或图像，直接作为一个块返回
        if len(group) == 1:
            element = group[0]
            is_table = element.get("metadata", {}).get("is_table", False)
            is_image = element.get("metadata", {}).get("is_image", False)
            
            if (is_table and self.preserve_tables) or (is_image and self.preserve_images):
                return [self._create_chunk_from_elements([element])]
        
        # 否则，根据大小限制进行分块
        chunks = []
        current_chunk_elements = []
        current_chunk_size = 0
        
        for element in group:
            content = element.get("content", "")
            content_size = len(content)
            
            # 如果单个元素超过最大块大小，需要单独处理
            if content_size > self.max_chunk_size:
                # 如果当前块不为空，先保存当前块
                if current_chunk_elements:
                    chunks.append(self._create_chunk_from_elements(current_chunk_elements))
                    current_chunk_elements = []
                    current_chunk_size = 0
                
                # 将大元素拆分并添加
                element_chunks = self._split_large_element(element)
                chunks.extend(element_chunks)
                continue
            
            # 如果添加当前元素后超过最大块大小，创建新块
            if current_chunk_size + content_size > self.max_chunk_size and current_chunk_elements:
                chunks.append(self._create_chunk_from_elements(current_chunk_elements))
                
                # 查找可以添加到新块的重叠元素
                overlap_elements = []
                overlap_size = 0
                for i in range(len(current_chunk_elements) - 1, -1, -1):
                    overlap_element = current_chunk_elements[i]
                    overlap_content_size = len(overlap_element.get("content", ""))
                    
                    if overlap_size + overlap_content_size <= self.chunk_overlap:
                        overlap_elements.insert(0, overlap_element)
                        overlap_size += overlap_content_size
                    else:
                        break
                
                current_chunk_elements = overlap_elements
                current_chunk_size = overlap_size
            
            # 将元素添加到当前块
            current_chunk_elements.append(element)
            current_chunk_size += content_size
        
        # 添加最后一个块
        if current_chunk_elements:
            chunks.append(self._create_chunk_from_elements(current_chunk_elements))
        
        return chunks
    
    def _split_large_element(self, element: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        拆分大型元素
        
        Args:
            element: 需要拆分的元素
            
        Returns:
            拆分后的块列表
        """
        content = element.get("content", "")
        metadata = element.get("metadata", {})
        
        # 不拆分表格和图像
        if metadata.get("is_table", False) or metadata.get("is_image", False):
            return [self._create_chunk_from_elements([element])]
        
        chunks = []
        start = 0
        content_length = len(content)
        
        while start < content_length:
            # 确定当前块的结束位置
            end = start + self.max_chunk_size
            
            if end >= content_length:
                # 最后一个块
                end = content_length
            else:
                # 寻找合适的断句点
                sentence_endings = [
                    content.rfind(". ", start, end),
                    content.rfind("。", start, end),
                    content.rfind("! ", start, end),
                    content.rfind("！", start, end),
                    content.rfind("? ", start, end),
                    content.rfind("？", start, end),
                    content.rfind("\n", start, end)
                ]
                
                # 过滤掉未找到的点位
                valid_endings = [e for e in sentence_endings if e > start]
                
                if valid_endings:
                    # 选择最靠近 end 的断句点
                    end = max(valid_endings) + 1
                else:
                    # 如果没有找到合适的断句点，尝试查找其他分隔符
                    separators = [
                        content.rfind(", ", start, end),
                        content.rfind("，", start, end),
                        content.rfind(";", start, end),
                        content.rfind("；", start, end),
                        content.rfind(" ", start, end)
                    ]
                    
                    valid_separators = [s for s in separators if s > start]
                    
                    if valid_separators:
                        end = max(valid_separators) + 1
                    # 如果都没有找到，就直接在 max_chunk_size 处截断
            
            # 创建新的元素对象
            chunk_content = content[start:end]
            chunk_element = {
                "type": element.get("type", ""),
                "content": chunk_content,
                "metadata": {**metadata, "is_split_chunk": True}
            }
            
            # 创建块并添加
            chunks.append(self._create_chunk_from_elements([chunk_element]))
            
            # 更新开始位置
            start = end
        
        return chunks
    
    def _create_chunk_from_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从元素列表创建文本块
        
        Args:
            elements: 元素列表
            
        Returns:
            文本块字典
        """
        if not elements:
            return {"content": "", "metadata": {}}
        
        # 提取块内容
        chunk_content = []
        chunk_elements = []
        
        # 记录块元数据
        chunk_types = []
        heading_paths = []
        tables_count = 0
        images_count = 0
        
        for element in elements:
            element_type = element.get("type", "")
            content = element.get("content", "")
            metadata = element.get("metadata", {})
            
            # 收集内容
            chunk_content.append(content)
            chunk_elements.append({
                "type": element_type,
                "length": len(content),
                "metadata": metadata
            })
            
            # 收集元数据
            chunk_types.append(element_type)
            
            # 记录标题路径
            heading_path = metadata.get("heading_path", [])
            if heading_path and heading_path not in heading_paths:
                heading_paths.append(heading_path)
            
            # 统计表格和图像
            if metadata.get("is_table", False):
                tables_count += 1
            if metadata.get("is_image", False):
                images_count += 1
        
        # 连接内容
        full_content = "\n".join(chunk_content)
        
        # 构建块元数据
        chunk_metadata = {
            "element_count": len(elements),
            "element_types": list(set(chunk_types)),
            "heading_paths": heading_paths,
            "tables_count": tables_count,
            "images_count": images_count,
            "elements": chunk_elements
        }
        
        return {
            "content": full_content,
            "metadata": chunk_metadata
        }
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: str) -> None:
        """
        将分块结果保存到文件
        
        Args:
            chunks: 文本块列表
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # 使用CustomJSONEncoder处理不可序列化的对象
                from main import CustomJSONEncoder
                json.dump(chunks, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
            logger.info(f"Saved {len(chunks)} chunks to {output_path}")
        except Exception as e:
            logger.error(f"Error saving chunks to {output_path}: {str(e)}")
            raise


# 测试代码
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 3:
        print("Usage: python text_chunker.py <input_json_file> <output_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        # 加载处理后的元素
        with open(input_file, 'r', encoding='utf-8') as f:
            elements = json.load(f)
        
        # 创建分块器并处理
        chunker = TextChunker(
            max_chunk_size=2000,
            min_chunk_size=100,
            chunk_overlap=200,
            filter_headers=True,  # 启用标题过滤
            main_content_only=True  # 只保留主要正文
        )
        
        chunks = chunker.create_chunks(elements)
        
        # 保存结果
        chunker.save_chunks(chunks, output_file)
        
        print(f"Successfully processed {len(elements)} elements into {len(chunks)} chunks")
        print(f"Output saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 