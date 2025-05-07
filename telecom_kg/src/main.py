"""
知识图谱构建主程序
"""

import os
import sys
import logging
import asyncio
import argparse
from typing import Optional
from logging.handlers import RotatingFileHandler

from utils.config import get_config_manager
from entity_extraction.extractor import extract_entities_from_directory
from relationship_extraction.extractor import extract_relationships_from_directory
from storage.neo4j_storage import store_knowledge_graph


# 配置日志
def setup_logging():
    """配置日志系统"""
    log_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file = os.path.join(log_dir, 'telecom_kg.log')
    
    # 创建日志处理器
    console_handler = logging.StreamHandler()
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    # 设置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    console_formatter = logging.Formatter(log_format)
    file_formatter = logging.Formatter(log_format)
    
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return root_logger

# 初始化日志
logger = setup_logging()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="通信领域知识图谱构建工具")
    
    parser.add_argument("--config", "-c", type=str, default=None,
                      help="配置文件路径，默认为config/settings.yaml")
    
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 实体提取命令
    extract_entities_parser = subparsers.add_parser("extract_entities", help="提取实体")
    extract_entities_parser.add_argument("--input", "-i", type=str, required=False,
                                       help="输入目录路径，默认使用配置文件中的路径")
    extract_entities_parser.add_argument("--output", "-o", type=str, required=False,
                                       help="输出目录路径，默认为output/entities")
    extract_entities_parser.add_argument("--target_file", "-t", type=str, required=False,
                                       help="指定要处理的文件名，仅处理这一个文件")
    
    # 关系提取命令
    extract_relationships_parser = subparsers.add_parser("extract_relationships", help="提取关系")
    extract_relationships_parser.add_argument("--entities", "-e", type=str, required=False,
                                           help="实体目录路径，默认为output/entities")
    extract_relationships_parser.add_argument("--input", "-i", type=str, required=False,
                                           help="原始文本目录路径，默认使用配置文件中的路径")
    extract_relationships_parser.add_argument("--output", "-o", type=str, required=False,
                                           help="输出目录路径，默认为output/relationships")
    extract_relationships_parser.add_argument("--target_file", "-t", type=str, required=False,
                                           help="指定要处理的文件名，仅处理这一个文件")
    
    # 知识图谱存储命令
    store_kg_parser = subparsers.add_parser("store_kg", help="存储知识图谱")
    store_kg_parser.add_argument("--entities", "-e", type=str, required=False,
                               help="实体目录路径，默认为output/entities")
    store_kg_parser.add_argument("--relationships", "-r", type=str, required=False,
                               help="关系目录路径，默认为output/relationships")
    
    # 全流程命令
    full_process_parser = subparsers.add_parser("full_process", help="执行完整流程")
    full_process_parser.add_argument("--input", "-i", type=str, required=False,
                                   help="输入目录路径，默认使用配置文件中的路径")
    full_process_parser.add_argument("--target_file", "-t", type=str, required=False,
                                   help="指定要处理的文件名，仅处理这一个文件")
    
    return parser.parse_args()


async def extract_entities(input_dir: str, output_dir: str, config_path: Optional[str] = None):
    """提取实体的流程"""
    logger.info(f"开始提取实体，输入目录: {input_dir}, 输出目录: {output_dir}")
    
    try:
        await extract_entities_from_directory(input_dir, output_dir, config_path)
        logger.info("实体提取完成")
        return True
    except Exception as e:
        logger.error(f"实体提取失败: {str(e)}")
        return False


async def extract_relationships(entity_dir: str, output_dir: str, input_dir: str, config_path: Optional[str] = None):
    """提取关系的流程"""
    logger.info(f"开始提取关系，实体目录: {entity_dir}, 输入目录: {input_dir}, 输出目录: {output_dir}")
    
    try:
        await extract_relationships_from_directory(entity_dir, output_dir, input_dir, config_path)
        logger.info("关系提取完成")
        return True
    except Exception as e:
        logger.error(f"关系提取失败: {str(e)}")
        return False


def store_kg(entity_dir: str, relationship_dir: str, config_path: Optional[str] = None):
    """存储知识图谱的流程"""
    logger.info(f"开始存储知识图谱，实体目录: {entity_dir}, 关系目录: {relationship_dir}")
    
    try:
        entity_count, relationship_count = store_knowledge_graph(entity_dir, relationship_dir, config_path)
        logger.info(f"知识图谱存储完成，共 {entity_count} 个实体和 {relationship_count} 个关系")
        return True
    except Exception as e:
        logger.error(f"知识图谱存储失败: {str(e)}")
        return False


async def full_process(input_dir: str, config_path: Optional[str] = None):
    """完整流程"""
    logger.info(f"开始执行完整知识图谱构建流程，输入目录: {input_dir}")
    
    # 获取配置
    config = get_config_manager(config_path)
    output_dir = config.get_path("output_dir")
    
    # 创建目录
    entity_dir = os.path.join(output_dir, "entities")
    relationship_dir = os.path.join(output_dir, "relationships")
    os.makedirs(entity_dir, exist_ok=True)
    os.makedirs(relationship_dir, exist_ok=True)
    
    # 执行实体提取
    logger.info("第一步: 提取实体")
    if not await extract_entities(input_dir, entity_dir, config_path):
        logger.error("实体提取失败，流程终止")
        return False
    
    # 执行关系提取
    logger.info("第二步: 提取关系")
    if not await extract_relationships(entity_dir, relationship_dir, input_dir, config_path):
        logger.error("关系提取失败，流程终止")
        return False
    
    # 存储知识图谱
    logger.info("第三步: 存储知识图谱")
    if not store_kg(entity_dir, relationship_dir, config_path):
        logger.error("知识图谱存储失败，流程终止")
        return False
    
    logger.info("知识图谱构建流程成功完成")
    return True


async def main():
    """主函数"""
    args = parse_args()
    
    config_path = args.config
    
    # 获取配置管理器
    config = get_config_manager(config_path)
    
    # 解析输入和输出路径
    if args.command == "extract_entities":
        input_dir = args.input or config.get_path("input_dir")
        output_dir = args.output or os.path.join(config.get_path("output_dir"), "entities")
        target_filename = args.target_file  # 获取目标文件名
        
        logger.info(f"开始提取实体，输入目录: {input_dir}, 输出目录: {output_dir}")
        if target_filename:
            logger.info(f"目标文件: {target_filename}")
        
        try:
            await extract_entities_from_directory(input_dir, output_dir, config_path, target_filename=target_filename)
            logger.info("实体提取完成")
            return True
        except Exception as e:
            logger.error(f"实体提取失败: {str(e)}")
            return False
            
    elif args.command == "extract_relationships":
        entity_dir = args.entities or os.path.join(config.get_path("output_dir"), "entities")
        input_dir = args.input or config.get_path("input_dir")
        output_dir = args.output or os.path.join(config.get_path("output_dir"), "relationships")
        target_filename = args.target_file  # 获取目标文件名
        
        logger.info(f"开始提取关系，实体目录: {entity_dir}, 输入目录: {input_dir}, 输出目录: {output_dir}")
        if target_filename:
            logger.info(f"目标文件: {target_filename}")
        
        try:
            await extract_relationships_from_directory(entity_dir, output_dir, input_dir, config_path, target_filename=target_filename)
            logger.info("关系提取完成")
            return True
        except Exception as e:
            logger.error(f"关系提取失败: {str(e)}")
            return False
        
    elif args.command == "store_kg":
        entity_dir = args.entities or os.path.join(config.get_path("output_dir"), "entities")
        relationship_dir = args.relationships or os.path.join(config.get_path("output_dir"), "relationships")
        
        success = store_kg(entity_dir, relationship_dir, config_path)
        sys.exit(0 if success else 1)
        
    elif args.command == "full_process":
        input_dir = args.input or config.get_path("input_dir")
        target_filename = args.target_file  # 获取目标文件名
        
        if target_filename:
            logger.info(f"目标文件: {target_filename}")
            # 创建目录
            config = get_config_manager(config_path)
            output_dir = config.get_path("output_dir")
            entity_dir = os.path.join(output_dir, "entities")
            relationship_dir = os.path.join(output_dir, "relationships")
            os.makedirs(entity_dir, exist_ok=True)
            os.makedirs(relationship_dir, exist_ok=True)
            
            # 只处理单个文件的完整流程
            logger.info("第一步: 提取实体")
            if not await extract_entities_from_directory(input_dir, entity_dir, config_path, target_filename=target_filename):
                logger.error("实体提取失败，流程终止")
                return False
                
            logger.info("第二步: 提取关系")
            if not await extract_relationships_from_directory(entity_dir, relationship_dir, input_dir, config_path, target_filename=target_filename):
                logger.error("关系提取失败，流程终止")
                return False
                
            logger.info("第三步: 存储知识图谱")
            if not store_kg(entity_dir, relationship_dir, config_path):
                logger.error("知识图谱存储失败，流程终止")
                return False
                
            logger.info("单文件知识图谱构建流程成功完成")
            return True
        else:
            success = await full_process(input_dir, config_path)
            sys.exit(0 if success else 1)
        
    else:
        print("请指定要执行的命令")
        print("可用命令: extract_entities, extract_relationships, store_kg, full_process")
        sys.exit(1)
    


if __name__ == "__main__":
    asyncio.run(main()) 