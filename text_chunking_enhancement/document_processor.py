#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文档处理器模块
负责将原始文档解析为结构化元素，包括文本、表格和图像
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import base64
import re

# 第三方库依赖
try:
    from unstructured.partition.auto import partition
    from unstructured.partition.html import partition_html
    from unstructured.partition.md import partition_md
    from unstructured.documents.elements import (
        Element, 
        Text, 
        Title, 
        NarrativeText, 
        ListItem,
        Table, 
        Image,
        PageBreak,
    )
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logging.warning("Unstructured.io library not found. Install with 'pip install unstructured'")

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文档处理器类，用于解析文档并提取结构化元素"""
    
    def __init__(self, use_ocr: bool = True, extract_images: bool = True):
        """
        初始化文档处理器
        
        Args:
            use_ocr: 是否使用OCR处理图像中的文本
            extract_images: 是否提取文档中的图像
        """
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError("Unstructured.io is required. Install with 'pip install unstructured'")
            
        self.use_ocr = use_ocr
        self.extract_images = extract_images
        logger.info("DocumentProcessor initialized with OCR=%s, extract_images=%s", 
                  use_ocr, extract_images)
    
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """
        处理文档并返回结构化元素列表
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            包含结构化元素的字典列表，每个字典包含元素类型、内容和元数据
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing document: {file_path}")
        
        # 根据文件类型选择合适的解析方法
        file_ext = file_path.suffix.lower()
        
        try:
            if file_ext == '.md':
                elements = partition_md(filename=str(file_path))
            elif file_ext == '.html' or file_ext == '.htm':
                elements = partition_html(filename=str(file_path))
            else:
                # 使用自动分区器处理其他类型
                extra_kwargs = {}
                if self.use_ocr:
                    extra_kwargs["ocr_languages"] = ["chi_sim+eng"]  # 支持中文和英文OCR
                
                elements = partition(
                    filename=str(file_path),
                    extract_images=self.extract_images,
                    **extra_kwargs
                )
            
            # 将元素转换为统一的字典格式
            processed_elements = self._process_elements(elements, file_path)
            
            logger.info(f"Successfully processed {len(processed_elements)} elements from {file_path}")
            return processed_elements
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def _process_elements(self, elements: List[Element], file_path: Path) -> List[Dict[str, Any]]:
        """
        处理元素列表，提取元数据和内容
        
        Args:
            elements: Unstructured库解析出的元素列表
            file_path: 原始文件路径
            
        Returns:
            处理后的元素字典列表
        """
        processed_elements = []
        current_heading_path = []
        
        for i, element in enumerate(elements):
            element_type = element.__class__.__name__
            element_dict = {
                "type": element_type,
                "content": str(element),
                "metadata": {
                    "source_file": str(file_path),
                    "element_id": i,
                }
            }
            
            # 提取并处理元素元数据
            if hasattr(element, "metadata"):
                # 修复：为ElementMetadata添加兼容性检查
                if hasattr(element.metadata, "items"):
                    for key, value in element.metadata.items():
                        if key not in ["filename", "filetype"]:  # 避免重复
                            element_dict["metadata"][key] = value
                else:
                    # 对于新版本的unstructured库，metadata可能不支持items()方法
                    # 尝试直接访问常见的属性
                    metadata_attrs = ["filename", "filetype", "page_number", "languages", "coordinates", "heading_level"]
                    for attr in metadata_attrs:
                        if hasattr(element.metadata, attr):
                            value = getattr(element.metadata, attr)
                            if value is not None:
                                element_dict["metadata"][attr] = value
            
            # 特殊处理不同类型的元素
            if isinstance(element, Title):
                # 处理标题路径
                heading_level = 1
                if hasattr(element.metadata, "get"):
                    heading_level = element.metadata.get("heading_level", 1)
                elif hasattr(element.metadata, "heading_level"):
                    heading_level = getattr(element.metadata, "heading_level", 1)
                
                # 根据标题级别更新当前标题路径
                while len(current_heading_path) >= heading_level:
                    current_heading_path.pop()
                current_heading_path.append(str(element))
                
                element_dict["metadata"]["heading_level"] = heading_level
                element_dict["metadata"]["heading_path"] = current_heading_path.copy()
                
            elif isinstance(element, NarrativeText) or isinstance(element, ListItem):
                # 为文本元素添加当前标题路径
                element_dict["metadata"]["heading_path"] = current_heading_path.copy()
                
            elif isinstance(element, Table):
                # 为表格添加特殊处理
                element_dict["metadata"]["heading_path"] = current_heading_path.copy()
                
                # 将表格内容转换为Markdown格式
                table_md = self._convert_table_to_markdown(element)
                element_dict["content"] = table_md
                element_dict["metadata"]["is_table"] = True
                element_dict["metadata"]["original_text"] = str(element)
                
            elif isinstance(element, Image) and self.extract_images:
                # 处理图像元素
                element_dict["metadata"]["heading_path"] = current_heading_path.copy()
                element_dict["metadata"]["is_image"] = True
                
                if hasattr(element, "image") and element.image:
                    # 将图像转换为base64编码
                    image_data = self._image_to_base64(element.image)
                    element_dict["metadata"]["image_data"] = image_data
                    
                # 保留图像描述文本
                image_text = None
                if hasattr(element.metadata, "get"):
                    image_text = element.metadata.get("image_text")
                elif hasattr(element.metadata, "image_text"):
                    image_text = getattr(element.metadata, "image_text")
                
                if image_text:
                    element_dict["metadata"]["image_text"] = image_text
            
            processed_elements.append(element_dict)
        
        return processed_elements
    
    def _convert_table_to_markdown(self, table_element: Table) -> str:
        """
        将表格元素转换为Markdown格式
        
        Args:
            table_element: 表格元素对象
            
        Returns:
            Markdown格式的表格字符串
        """
        # 修复：为ElementMetadata添加兼容性检查
        has_html = False
        if hasattr(table_element, "metadata"):
            if hasattr(table_element.metadata, "get"):
                has_html = table_element.metadata.get("text_as_html")
            elif hasattr(table_element.metadata, "text_as_html"):
                has_html = getattr(table_element.metadata, "text_as_html")
        
        if not hasattr(table_element, "metadata") or not has_html:
            # 如果没有HTML表示，则尝试通过其他方式构建Markdown表格
            return str(table_element)
        
        try:
            # 获取HTML内容
            html_content = ""
            if hasattr(table_element.metadata, "get"):
                html_content = table_element.metadata.get("text_as_html", "")
            elif hasattr(table_element.metadata, "text_as_html"):
                html_content = getattr(table_element.metadata, "text_as_html", "")
            
            # 从HTML中提取表格行
            rows = re.findall(r'<tr>(.*?)</tr>', html_content, re.DOTALL)
            
            if not rows:
                return str(table_element)
            
            markdown_rows = []
            header_processed = False
            
            for row in rows:
                # 检测是否为表头行
                if '<th' in row and not header_processed:
                    # 提取表头单元格
                    headers = re.findall(r'<th.*?>(.*?)</th>', row, re.DOTALL)
                    if headers:
                        markdown_rows.append('| ' + ' | '.join(headers) + ' |')
                        # 添加分隔行
                        markdown_rows.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
                        header_processed = True
                else:
                    # 提取数据单元格
                    cells = re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL)
                    if cells:
                        markdown_rows.append('| ' + ' | '.join(cells) + ' |')
            
            # 如果没有找到表头，则将第一行作为表头
            if not header_processed and markdown_rows:
                first_row = markdown_rows[0]
                col_count = first_row.count('|') - 1
                markdown_rows.insert(1, '| ' + ' | '.join(['---'] * col_count) + ' |')
            
            return '\n'.join(markdown_rows)
            
        except Exception as e:
            logger.warning(f"Error converting table to Markdown: {str(e)}")
            return str(table_element)
    
    def _image_to_base64(self, image) -> str:
        """
        将图像对象转换为base64编码字符串
        
        Args:
            image: 图像对象
            
        Returns:
            base64编码的图像字符串
        """
        import io
        
        try:
            # 将图像转换为字节流
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format if image.format else 'PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # 转换为base64编码
            return base64.b64encode(img_byte_arr).decode('utf-8')
        except Exception as e:
            logger.warning(f"Error converting image to base64: {str(e)}")
            return ""


# 测试代码
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python document_processor.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    processor = DocumentProcessor()
    
    try:
        elements = processor.process(file_path)
        print(f"Processed {len(elements)} elements from {file_path}")
        
        # 打印前5个元素作为示例
        for i, element in enumerate(elements[:5]):
            print(f"\nElement {i+1}:")
            print(f"Type: {element['type']}")
            print(f"Content (preview): {element['content'][:100]}...")
            print(f"Metadata: {element['metadata']}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 