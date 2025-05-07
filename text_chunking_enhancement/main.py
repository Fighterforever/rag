#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强型文本块划分系统的主程序
提供命令行界面，集成文档处理和智能分块功能
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from document_processor import DocumentProcessor
from text_chunker import TextChunker

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("text_chunking.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加JSON序列化处理函数
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # 处理不可序列化的对象
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, '__str__'):
            return str(obj)
        else:
            return repr(obj)

def process_single_file(
    file_path: str, 
    output_dir: str, 
    settings: Dict[str, Any]
) -> Dict[str, Any]:
    """
    处理单个文件
    
    Args:
        file_path: 文件路径
        output_dir: 输出目录
        settings: 处理设置
        
    Returns:
        处理结果信息
    """
    try:
        file_path_obj = Path(file_path)
        file_name = file_path_obj.name
        
        # 创建输出目录
        output_path = Path(output_dir) / file_name
        os.makedirs(output_path.parent, exist_ok=True)
        
        # 1. 使用文档处理器解析文档
        processor = DocumentProcessor(
            use_ocr=settings.get('use_ocr', True),
            extract_images=settings.get('extract_images', True)
        )
        
        elements = processor.process(file_path_obj)
        
        # 保存解析结果
        elements_file = output_path.with_suffix('.elements.json')
        with open(elements_file, 'w', encoding='utf-8') as f:
            json.dump(elements, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
        
        # 2. 使用分块器创建文本块
        chunker = TextChunker(
            max_chunk_size=settings.get('max_chunk_size', 2000),
            min_chunk_size=settings.get('min_chunk_size', 100),
            chunk_overlap=settings.get('chunk_overlap', 200),
            preserve_tables=settings.get('preserve_tables', True),
            preserve_images=settings.get('preserve_images', True),
            respect_headers=settings.get('respect_headers', True),
            gemini_api_key=settings.get('gemini_api_key'),
            filter_headers=settings.get('filter_headers', True),
            main_content_only=settings.get('main_content_only', True)
        )
        
        chunks = chunker.create_chunks(elements, original_file_path=file_path_obj)
        
        # 保存分块结果
        chunks_file = output_path.with_suffix('.chunks.json')
        chunker.save_chunks(chunks, chunks_file)
        
        return {
            "file": str(file_path_obj),
            "element_count": len(elements),
            "chunk_count": len(chunks),
            "elements_file": str(elements_file),
            "chunks_file": str(chunks_file),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "file": str(file_path),
            "status": "error",
            "error": str(e)
        }

def process_directory(
    input_dir: str, 
    output_dir: str, 
    settings: Dict[str, Any],
    file_types: Optional[List[str]] = None,
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    处理目录中的所有文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        settings: 处理设置
        file_types: 要处理的文件类型列表
        max_workers: 最大并行处理数
        
    Returns:
        处理结果列表
    """
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist or is not a directory")
    
    # 默认文件类型
    if file_types is None:
        file_types = [".md", ".pdf", ".docx", ".html", ".txt"]
    
    # 查找所有符合条件的文件
    files_to_process = []
    for file_type in file_types:
        files_to_process.extend(input_dir.glob(f"**/*{file_type}"))
    
    if not files_to_process:
        logger.warning(f"No files found with types {file_types} in {input_dir}")
        return []
    
    logger.info(f"Found {len(files_to_process)} files to process")
    
    # 并行处理文件
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for file_path in files_to_process:
            relative_path = file_path.relative_to(input_dir)
            future = executor.submit(
                process_single_file, 
                str(file_path), 
                os.path.join(output_dir, str(relative_path.parent)), 
                settings
            )
            futures[future] = file_path
        
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed processing {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                results.append({
                    "file": str(file_path),
                    "status": "error",
                    "error": str(e)
                })
    
    return results

def main():
    """主函数，解析命令行参数并执行处理"""
    parser = argparse.ArgumentParser(description="增强型文本块划分系统")
    
    # 基本参数
    parser.add_argument("--input", required=True, help="输入文件或目录路径")
    parser.add_argument("--output", required=True, help="输出目录路径")
    
    # 处理单个文件还是目录
    parser.add_argument("--batch", action="store_true", help="批量处理目录")
    
    # 文档处理器选项
    parser.add_argument("--use-ocr", action="store_true", help="启用OCR处理")
    parser.add_argument("--extract-images", action="store_true", help="提取并处理图像")
    
    # 分块器选项
    parser.add_argument("--max-chunk-size", type=int, default=2000, help="最大块大小（字符数）")
    parser.add_argument("--min-chunk-size", type=int, default=100, help="最小块大小（字符数）")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="块重叠（字符数）")
    parser.add_argument("--no-preserve-tables", action="store_true", help="不保留表格完整性")
    parser.add_argument("--no-preserve-images", action="store_true", help="不保留图像完整性")
    parser.add_argument("--no-respect-headers", action="store_true", help="不使用标题作为分块边界")
    parser.add_argument("--no-filter-headers", action="store_true", help="不过滤无关标题")
    parser.add_argument("--no-main-content-only", action="store_true", help="不限制只处理主要内容")
    
    # 批处理选项
    parser.add_argument("--file-types", nargs="+", help="要处理的文件类型列表，例如 .md .pdf")
    parser.add_argument("--max-workers", type=int, default=4, help="最大并行处理数")
    
    # 新增：Gemini API 密钥
    parser.add_argument("--gemini-api-key", type=str, required=True, help="Google Gemini API 密钥")
    
    args = parser.parse_args()
    
    # 整合设置
    settings = {
        "use_ocr": args.use_ocr,
        "extract_images": args.extract_images,
        "max_chunk_size": args.max_chunk_size,
        "min_chunk_size": args.min_chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "preserve_tables": not args.no_preserve_tables,
        "preserve_images": not args.no_preserve_images,
        "respect_headers": not args.no_respect_headers,
        "filter_headers": not args.no_filter_headers,
        "main_content_only": not args.no_main_content_only,
        "gemini_api_key": args.gemini_api_key
    }
    
    try:
        # 确保输出目录存在
        os.makedirs(args.output, exist_ok=True)
        
        # 记录配置信息
        logger.info(f"Input: {args.input}")
        logger.info(f"Output: {args.output}")
        # 不记录完整的 settings，避免 API Key 泄露到日志
        log_settings = settings.copy()
        if "gemini_api_key" in log_settings:
            log_settings["gemini_api_key"] = "****" # Mask API Key
        logger.info(f"Settings: {log_settings}")
        
        # 处理文件或目录
        if args.batch:
            # 批量处理目录
            file_types = args.file_types if args.file_types else None
            results = process_directory(
                args.input, 
                args.output, 
                settings,
                file_types,
                args.max_workers
            )
            
            # 保存处理摘要
            summary_file = os.path.join(args.output, "processing_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Processing complete. Summary saved to {summary_file}")
            
            # 打印摘要统计
            success_count = sum(1 for r in results if r.get("status") == "success")
            error_count = sum(1 for r in results if r.get("status") == "error")
            total_elements = sum(r.get("element_count", 0) for r in results if r.get("status") == "success")
            total_chunks = sum(r.get("chunk_count", 0) for r in results if r.get("status") == "success")
            
            print(f"处理完成: 成功 {success_count} 个文件, 失败 {error_count} 个文件")
            print(f"共处理 {total_elements} 个元素, 生成 {total_chunks} 个文本块")
            print(f"摘要已保存至 {summary_file}")
            
        else:
            # 处理单个文件
            result = process_single_file(args.input, args.output, settings)
            
            if result["status"] == "success":
                print(f"处理完成: {result['file']}")
                print(f"共处理 {result['element_count']} 个元素, 生成 {result['chunk_count']} 个文本块")
                print(f"解析结果: {result['elements_file']}")
                print(f"分块结果: {result['chunks_file']}")
            else:
                print(f"处理失败: {result['file']}")
                print(f"错误: {result['error']}")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        # 添加 traceback 以便调试
        import traceback
        logger.error(traceback.format_exc())
        print(f"发生意外错误: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 