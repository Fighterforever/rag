#!/usr/bin/env python3
import sys
import os
import logging
import traceback

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径 (假设脚本在telecom_kg目录下运行)
sys.path.insert(0, '.')

logger.info('启动导入脚本...')

try:
    # 导入Neo4j存储类
    from src.storage.neo4j_storage import Neo4jStorage
    logger.info('成功导入 Neo4jStorage')

    # 定义源数据目录 (相对于telecom_kg)
    entity_dir = 'output/gemini_test/entities'
    relationship_dir = 'output/gemini_test/relationships'

    logger.info(f'实体目录: {os.path.abspath(entity_dir)}')
    logger.info(f'关系目录: {os.path.abspath(relationship_dir)}')

    # 检查目录是否存在
    if not os.path.exists(entity_dir):
        logger.error('错误: 实体目录不存在!')
        sys.exit(1)
    if not os.path.exists(relationship_dir):
        logger.error('错误: 关系目录不存在!')
        sys.exit(1)

    logger.info('准备实例化 Neo4jStorage...')
    storage = Neo4jStorage()

    logger.info('开始导入，将首先清空数据库...')
    # 处理清空、索引和导入
    entity_count, relation_count = storage.store_knowledge_graph(entity_dir, relationship_dir)
    storage.close()

    print(f'导入成功! 清空数据库后，导入了 {entity_count} 个实体和 {relation_count} 个关系。')
    logger.info(f'导入成功! 清空数据库后，导入了 {entity_count} 个实体和 {relation_count} 个关系。')

except ImportError as e:
    logger.error(f'ImportError: {e}')
    logger.error('请确认Neo4jStorage模块及其依赖项已正确安装，并且在 telecom_kg/src/storage/ 目录下')
    sys.exit(1)
except Exception as e:
    logger.error(f'导入过程中发生错误: {e}')
    logger.error(traceback.format_exc())
    sys.exit(1)

logger.info('导入脚本执行完毕.')
