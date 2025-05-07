"""
配置加载和管理模块
"""

import os
import yaml
import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """配置管理器，负责加载和管理配置"""
    
    def __init__(self, config_path: str):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._resolve_env_variables()  # 解析环境变量
        self.cache = {}  # 用于缓存LLM响应
        
        # 启用调试模式，如果配置中指定
        self._enable_debug_mode()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            return {}
    
    def _resolve_env_variables(self):
        """解析配置中的环境变量引用"""
        def _process_value(value):
            if isinstance(value, str):
                # 处理 ${ENV_VAR} 格式的环境变量引用
                env_var_pattern = r'\${([A-Za-z0-9_]+)}'
                matches = re.findall(env_var_pattern, value)
                
                if matches:
                    for match in matches:
                        env_value = os.environ.get(match, '')
                        value = value.replace(f'${{{match}}}', env_value)
                
                # 处理 os.environ.get("ENV_VAR", "") 格式
                if value.startswith('os.environ.get('):
                    env_var_pattern = r'os\.environ\.get\(["\']([A-Za-z0-9_]+)["\'],\s*["\'](.*?)["\']\)'
                    match = re.search(env_var_pattern, value)
                    if match:
                        env_name, default = match.groups()
                        value = os.environ.get(env_name, default)
                        
                return value
            elif isinstance(value, dict):
                return {k: _process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_process_value(item) for item in value]
            else:
                return value
        
        self.config = _process_value(self.config)
        
    def get(self, key_path: str, default=None) -> Any:
        """
        根据路径获取配置值
        
        Args:
            key_path: 配置键路径，使用点分隔，如 "models.openai.api_key"
            default: 默认值，如果路径不存在则返回此值
            
        Returns:
            配置值或默认值
        """
        parts = key_path.split('.')
        value = self.config
        
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_path(self, key: str) -> str:
        """
        获取路径配置项
        
        Args:
            key: 路径配置键，如 "input_dir"
            
        Returns:
            路径字符串
        """
        path = self.get(f"paths.{key}")
        if not path:
            logger.warning(f"未找到路径配置: {key}")
            return ""
        
        # 处理 ~ 表示的用户目录
        path = os.path.expanduser(path)
        
        # 创建目录如果不存在
        if not os.path.exists(path) and any(key.endswith(suffix) for suffix in ["_dir", "directory"]):
            try:
                os.makedirs(path, exist_ok=True)
                logger.info(f"创建目录: {path}")
            except Exception as e:
                logger.error(f"创建目录失败: {str(e)}")
        
        return path
    
    def get_model_config(self, model_id: str) -> Dict[str, Any]:
        """
        获取模型配置
        
        Args:
            model_id: 模型ID，如 "default_chat_model" 或 "qwen-max"
            
        Returns:
            模型配置字典
        """
        # 尝试从models中直接获取
        model_config = self.get(f"models.{model_id}")
        
        if not model_config:
            # 尝试根据active_model获取
            active_model = self.get("active_model")
            model_config = self.get(f"models.{active_model}")
            
            if not model_config:
                # 找不到，使用默认模型
                model_config = self.get("models.default_chat_model", {})
                logger.warning(f"未找到模型配置 {model_id}，使用默认模型")
        
        return model_config or {}
    
    def get_entity_types(self) -> list:
        """
        获取实体类型列表
        
        Returns:
            实体类型列表
        """
        if "entity_extraction" not in self.config:
            raise ValueError("配置中未找到entity_extraction配置项")
        
        if "entity_types" not in self.config["entity_extraction"]:
            raise ValueError("配置中未找到entity_types配置项")
        
        return self.config["entity_extraction"]["entity_types"]
    
    def get_relation_types(self) -> list:
        """
        获取关系类型列表
        
        Returns:
            关系类型列表
        """
        if "relationship_extraction" not in self.config:
            raise ValueError("配置中未找到relationship_extraction配置项")
        
        if "relation_types" not in self.config["relationship_extraction"]:
            raise ValueError("配置中未找到relation_types配置项")
        
        return self.config["relationship_extraction"]["relation_types"]

    def _enable_debug_mode(self):
        """启用调试模式，调整日志级别"""
        # 如果日志级别是DEBUG，调整之前的所有handlers级别为DEBUG
        if self.get("logging.level", "INFO") == "DEBUG":
            loggers = [logging.getLogger()]  # 根日志器
            loggers.extend([logging.getLogger(name) for name in logging.root.manager.loggerDict])
            
            for logger in loggers:
                for handler in logger.handlers:
                    handler.setLevel(logging.DEBUG)
                logger.setLevel(logging.DEBUG)
                
            logging.info("调试模式已启用，所有日志级别已设为DEBUG")


# 单例配置管理器
_config_manager = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    获取配置管理器单例
    
    Args:
        config_path: 配置文件路径，仅首次调用时有效
        
    Returns:
        配置管理器实例
    """
    global _config_manager
    
    if _config_manager is None:
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'config',
                'settings.yaml'
            )
        _config_manager = ConfigManager(config_path)
    
    return _config_manager 