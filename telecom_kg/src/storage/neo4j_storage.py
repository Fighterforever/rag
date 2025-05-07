"""
Neo4j 知识图谱存储模块
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from neo4j import GraphDatabase, Driver

from src.utils.config import get_config_manager


logger = logging.getLogger(__name__)

class Neo4jStorage:
    """Neo4j 知识图谱存储类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化 Neo4j 存储
        
        Args:
            config_path: 配置文件路径，如不提供则使用默认路径
        """
        self.config = get_config_manager(config_path)
        neo4j_config = self.config.get("neo4j")
        
        if not neo4j_config:
            raise ValueError("未找到 Neo4j 配置")
        
        self.uri = neo4j_config.get("uri")
        self.username = neo4j_config.get("username")
        self.password = neo4j_config.get("password")
        self.database = neo4j_config.get("database", "neo4j")
        
        if not self.uri or not self.username or not self.password:
            raise ValueError("Neo4j 配置不完整")
        
        self.driver = None
    
    def connect(self) -> Driver:
        """
        连接 Neo4j 数据库
        
        Returns:
            Neo4j 驱动
        """
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.username, self.password)
                )
                logger.info(f"已连接到 Neo4j 数据库: {self.uri}")
            except Exception as e:
                logger.error(f"连接 Neo4j 数据库失败: {str(e)}")
                raise
        
        return self.driver
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("已关闭 Neo4j 数据库连接")
    
    def store_entities(self, entities: List[Dict[str, Any]]) -> int:
        """
        存储实体列表到Neo4j
        
        Args:
            entities: 实体列表
            
        Returns:
            成功存储的实体数量
        """
        driver = self.connect()
        count = 0
        
        try:
            with driver.session(database=self.database) as session:
                for entity in entities:
                    # 标准化 entity_type: 替换空格为下划线，并移除其他非字母数字字符
                    original_type = entity.get('entity_type', 'Unknown')
                    # 移除非字母数字下划线的字符，并将空格替换为下划线
                    sanitized_type = re.sub(r'[^a-zA-Z0-9_]', '', original_type.replace(' ', '_'))
                    # 确保处理后的类型不为空，如果为空则设置为 Unknown
                    if not sanitized_type:
                        sanitized_type = 'Unknown'
                        
                    if original_type != sanitized_type:
                        logger.info(f"Sanitizing entity type: '{original_type}' -> '{sanitized_type}'")

                    # 构建属性字典
                    properties = {
                        'name': entity['entity_name'],
                        'description': entity.get('entity_description', ''),
                        'aliases': json.dumps(entity.get('entity_aliases', []), ensure_ascii=False),
                        'original_type': original_type # 可以选择性保留原始类型
                    }
                    
                    # 使用处理后的类型作为节点标签
                    # 同时添加一个通用的 :Entity 标签，方便查询
                    tx_result = session.write_transaction(
                        self._create_entity_node, sanitized_type, properties
                    )
                    if tx_result: # 假设 _create_entity_node 成功时返回 True 或非 None
                       count += 1
                       
        except Exception as e:
            logger.error(f"存储实体失败: {str(e)}")
            # 可以考虑抛出异常或返回部分成功计数
            # raise # 如果希望失败时停止整个流程
            
        return count
    
    def store_relationships(self, relationships: List[Dict[str, Any]]) -> int:
        """
        存储关系到 Neo4j
        
        Args:
            relationships: 关系列表
            
        Returns:
            存储成功的关系数量
        """
        if not relationships:
            return 0
            
        driver = self.connect()
        
        try:
            with driver.session(database=self.database) as session:
                # 创建关系
                count = 0
                for relation in relationships:
                    result = session.write_transaction(
                        self._create_relationship,
                        source_entity=relation["source_entity"],
                        target_entity=relation["target_entity"],
                        relation_type=relation["relation_type"],
                        description=relation["relation_description"],
                        strength=relation["relation_strength"],
                        source_type=relation.get("source_entity_type"),
                        target_type=relation.get("target_entity_type")
                    )
                    count += result
                
                return count
        except Exception as e:
            logger.error(f"存储关系失败: {str(e)}")
            raise
    
    def store_knowledge_graph(self, entity_dir: str, relationship_dir: str) -> Tuple[int, int]:
        """
        从目录中存储知识图谱
        
        Args:
            entity_dir: 实体目录
            relationship_dir: 关系目录
            
        Returns:
            (存储的实体数量, 存储的关系数量)
        """
        # 连接数据库
        driver = self.connect()
        
        # 清空数据库
        self.clear_database()
        
        # 创建索引
        self._create_indexes()
        
        # 获取所有实体文件
        entity_files = []
        for root, _, filenames in os.walk(entity_dir):
            for filename in filenames:
                if filename.endswith("_entities.json") and not filename.startswith('.'):
                    entity_files.append(os.path.join(root, filename))
        
        logger.info(f"找到 {len(entity_files)} 个实体文件")
        
        # 存储所有实体
        total_entities = 0
        for entity_file in entity_files:
            try:
                with open(entity_file, 'r', encoding='utf-8') as f:
                    entities = json.load(f)
                
                if entities:
                    count = self.store_entities(entities)
                    total_entities += count
                    logger.info(f"存储 {os.path.basename(entity_file)} 中的 {count} 个实体")
            except Exception as e:
                logger.error(f"处理实体文件 {entity_file} 失败: {str(e)}")
        
        # 获取所有关系文件
        relationship_files = []
        for root, _, filenames in os.walk(relationship_dir):
            for filename in filenames:
                if filename.endswith("_relationships.json") and not filename.startswith('.'):
                    relationship_files.append(os.path.join(root, filename))
        
        logger.info(f"找到 {len(relationship_files)} 个关系文件")
        
        # 存储所有关系
        total_relationships = 0
        for relationship_file in relationship_files:
            try:
                with open(relationship_file, 'r', encoding='utf-8') as f:
                    relationships = json.load(f)
                
                if relationships:
                    count = self.store_relationships(relationships)
                    total_relationships += count
                    logger.info(f"存储 {os.path.basename(relationship_file)} 中的 {count} 个关系")
            except Exception as e:
                logger.error(f"处理关系文件 {relationship_file} 失败: {str(e)}")
        
        logger.info(f"知识图谱存储完成，共 {total_entities} 个实体和 {total_relationships} 个关系")
        
        return total_entities, total_relationships
    
    def clear_database(self):
        """清空数据库"""
        driver = self.connect()
        
        try:
            with driver.session(database=self.database) as session:
                session.write_transaction(self._clear_db)
                logger.info("数据库已清空")
        except Exception as e:
            logger.error(f"清空数据库失败: {str(e)}")
            raise
    
    def _create_indexes(self):
        """创建索引"""
        driver = self.connect()
        
        try:
            with driver.session(database=self.database) as session:
                # 创建全局实体名称索引
                session.write_transaction(
                    self._create_constraint,
                    label="Entity",
                    property="name"
                )
                
                # 创建通用实体类型索引
                common_entity_types = [
                    "Organization", "OrganizationUnit", "Person", 
                    "System", "Device", "Technology", "Resource",
                    "Protocol", "Standard", "ManagementStandard", 
                    "Process", "Service", "Concept", "Location"
                ]
                
                for type_name in common_entity_types:
                    session.write_transaction(
                        self._create_index,
                        label=type_name,
                        property="name"
                    )
                
                logger.info("已创建索引")
        except Exception as e:
            logger.error(f"创建索引失败: {str(e)}")
            # 不抛出异常，因为索引可能已存在
            pass
    
    @staticmethod
    def _clear_db(tx):
        """清空数据库事务"""
        tx.run("MATCH (n) DETACH DELETE n")
        return 1
    
    @staticmethod
    def _create_index(tx, label, property):
        """创建索引事务"""
        tx.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{property})")
        return 1
    
    @staticmethod
    def _create_constraint(tx, label, property):
        """创建约束事务"""
        tx.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{property} IS UNIQUE")
        return 1
    
    @staticmethod
    def _create_entity_node(tx, entity_type: str, properties: Dict[str, Any]):
        """Neo4j事务函数：创建或合并实体节点"""
        # Cypher 查询，使用参数化查询防止注入
        # 使用处理后的 entity_type 作为标签
        # 添加 :Entity 标签便于通用查询
        query = (
            f"MERGE (e:Entity:{entity_type} {{name: $props.name}}) "
            "ON CREATE SET e = $props "
            "ON MATCH SET e += $props "
            "RETURN e"
        )
        result = tx.run(query, props=properties)
        return result.single() # 返回创建或匹配的节点，或 None
    
    @staticmethod
    def _create_relationship(tx, source_entity, target_entity, relation_type, 
                           description, strength, source_type=None, target_type=None):
        """创建关系事务"""
        # 格式化关系类型为Neo4j格式（去除空格，使用下划线）
        neo4j_rel_type = relation_type.replace(" ", "_").upper()
        
        # 构建查询
        query = """
        MATCH (source:Entity {name: $source_entity})
        MATCH (target:Entity {name: $target_entity})
        MERGE (source)-[r:%s]->(target)
        SET r.description = $description,
            r.strength = $strength,
            r.relation_type = $relation_type
        RETURN r
        """ % neo4j_rel_type
        
        result = tx.run(
            query,
            source_entity=source_entity,
            target_entity=target_entity,
            description=description,
            strength=strength,
            relation_type=relation_type
        )
        return result.single() is not None


def store_knowledge_graph(entity_dir: str, relationship_dir: str, config_path: Optional[str] = None) -> Tuple[int, int]:
    """
    将实体和关系存储到Neo4j知识图谱
    
    Args:
        entity_dir: 实体文件目录
        relationship_dir: 关系文件目录
        config_path: 配置文件路径
        
    Returns:
        (存储的实体数量, 存储的关系数量)
    """
    storage = Neo4jStorage(config_path)
    try:
        return storage.store_knowledge_graph(entity_dir, relationship_dir)
    finally:
        storage.close() 