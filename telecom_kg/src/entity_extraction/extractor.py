"""
实体提取模块
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
import aiohttp
# import tiktoken  # 注释掉直接导入

# Import Google Gemini API library (仍保留，但可能不使用)
import google.generativeai as genai

from utils.config import get_config_manager

logger = logging.getLogger(__name__)

# 创建一个简单的替代类用于Token计数
class SimpleTikToken:
    def __init__(self, encoding_name):
        self.encoding_name = encoding_name
        
    def encode(self, text):
        # 简单地按字符计数，每个中文字符算 2 个 token，英文单词算 1 个 token
        # 这只是一个粗略的估计，实际情况会有差异
        tokens = []
        for char in text:
            if ord(char) > 127:  # 中文和其他非ASCII字符
                tokens.extend([1, 2])  # 假设是 2 个 token
            else:
                tokens.append(1)  # 假设是 1 个 token
        return tokens

# 尝试导入 tiktoken，如果失败则使用简单的替代函数
try:
    import tiktoken
    tiktoken_available = True
    def get_encoding(encoding_name):
        try:
            return tiktoken.get_encoding(encoding_name)
        except:
            # 如果特定编码不可用，则使用一个基本的编码器
            logger.warning(f"编码 {encoding_name} 不可用，使用cl100k_base替代")
            return tiktoken.get_encoding("cl100k_base")
except Exception as e:
    tiktoken_available = False
    logger.warning(f"无法导入 tiktoken: {e}，将使用简单的字符计数替代")
    
    def get_encoding(encoding_name):
        return SimpleTikToken(encoding_name)

class EntityExtractor:
    """实体提取器，用于从文本中提取通信领域实体"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化实体提取器
        
        Args:
            config_path: 配置文件路径，如不提供则使用默认路径
        """
        self.config = get_config_manager(config_path)
        self.prompts_dir = self.config.get_path("prompts_dir")
        self.cache_dir = self.config.get_path("cache_dir")
        
        # 获取当前配置的活跃模型
        self.active_model_id = self.config.get("active_model", "default_chat_model")
        logger.info(f"使用模型: {self.active_model_id}")
        
        # 如果未指定活跃模型或指定为gemini，则使用Gemini
        self.use_gemini = self.active_model_id == "gemini"
        
        # 根据选择的模型初始化客户端
        self.gemini_model = None
        self.model_config = None
        
        if self.use_gemini:
            # 初始化Gemini客户端
            try:
                gemini_config = self.config.get("gemini")
                if not gemini_config or not gemini_config.get("api_key"):
                    logger.error("Gemini API key not found in configuration (gemini.api_key).")
                    raise ValueError("Missing Gemini API Key")
                
                genai.configure(api_key=gemini_config["api_key"])
                model_name = gemini_config.get("model_name", "gemini-1.5-pro-latest") # Default to 1.5 pro
                # Configure safety settings to be less restrictive if needed, example:
                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
                self.gemini_model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
                logger.info(f"Google Gemini client initialized successfully with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Google Gemini client: {e}")
                # Keep self.gemini_model as None, subsequent calls will fail
        else:
            # 使用OpenAI兼容API (默认模型或Qwen)
            self.model_config = self.config.get_model_config(self.active_model_id)
            logger.info(f"使用OpenAI兼容API: {self.model_config.get('model')} ({self.model_config.get('api_base')})")
            # 创建token计数器
            try:
                encoding_name = self.model_config.get("encoding_model", "cl100k_base")
                self.encoding = get_encoding(encoding_name)
                logger.info(f"成功初始化token计数器: {encoding_name}")
            except Exception as e:
                logger.warning(f"初始化token计数器失败: {e}，使用简单字符计数")
                self.encoding = SimpleTikToken(self.model_config.get("encoding_model", "cl100k_base"))
        
        # 加载提示模板
        prompt_path = os.path.join(self.prompts_dir, self.config.get("entity_extraction.prompt_template"))
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

    async def extract_entities(self, text: str, text_id: str) -> List[Dict[str, Any]]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            text_id: 文本ID，用于缓存
            
        Returns:
            提取出的实体列表
        """
        # 检查缓存
        cache_file = os.path.join(self.cache_dir, f"entity_{text_id}.json")
        if os.path.exists(cache_file) and self.config.get("processing.cache_enabled", True):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    logger.info(f"从缓存加载实体: {text_id}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载缓存失败: {str(e)}")
        
        # 构造提示
        prompt = self.prompt_template.replace("{input_text}", text)
        
        # 调用模型
        entities = await self._call_llm(prompt)
        
        # 后处理
        entities = self._post_process_entities(entities, text)
        
        # 保存缓存
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        
        return entities
    
    async def _call_llm(self, prompt: str) -> List[Dict[str, Any]]:
        """
        调用大模型提取实体
        
        Args:
            prompt: 发送给模型的完整提示
            
        Returns:
            从模型响应解析出的实体列表，失败则返回空列表
        """
        # 根据激活的模型选择调用方式
        if self.use_gemini and self.gemini_model:
            return await self._call_gemini(prompt)
        elif self.model_config:
            return await self._call_openai_compatible(prompt)
        else:
            logger.error("No valid model configuration available.")
            return []
    
    async def _call_openai_compatible(self, prompt: str) -> List[Dict[str, Any]]:
        """
        调用OpenAI兼容API(包括Qwen)
        
        Args:
            prompt: 提示文本
            
        Returns:
            解析后的实体列表
        """
        # 使用提示和模型ID创建缓存键
        cache_key = f"{self.active_model_id}_{hash(prompt)}"
        
        # 检查缓存
        if self.config.get("processing.cache_enabled", True) and cache_key in self.config.get("cache", {}):
            logger.info(f"使用缓存的响应，模型: {self.active_model_id}")
            return self.config.get("cache")[cache_key]
            
        result = None
        model_id = self.model_config.get("model", "").lower()
        api_key = self.model_config.get("api_key")
        base_url = self.model_config.get("api_base")
        
        try:
            # 获取并检查max_tokens参数是否符合模型限制
            max_tokens = self.model_config.get("max_tokens", 8192)
            
            # 针对不同模型的token限制进行调整
            if "qwen" in model_id and max_tokens > 8192:
                logger.warning(f"Qwen模型max_tokens参数超过限制(8192)，已自动调整为8192")
                max_tokens = 8192
            elif "chatglm" in model_id and max_tokens > 8192:
                logger.warning(f"ChatGLM模型max_tokens参数超过限制(8192)，已自动调整为8192")
                max_tokens = 8192
            elif "baichuan" in model_id and max_tokens > 4096:
                logger.warning(f"Baichuan模型max_tokens参数超过限制(4096)，已自动调整为4096")
                max_tokens = 4096
                
            # 准备API请求
            payload = {
                "model": self.model_config.get("model"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "top_p": 0.8,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }

            api_url = f"{base_url}/chat/completions"
            max_retries = self.model_config.get("max_retries", 5)
            retry_count = 0
            response_text = ""

            # 执行API调用，带重试
            while retry_count <= max_retries:
                try:
                    async with aiohttp.ClientSession() as session:
                        logger.info(f"调用API: {api_url}, 模型: {model_id}, max_tokens: {max_tokens}")
                        async with session.post(api_url, headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}, json=payload, timeout=300) as response:
                            if response.status == 200:
                                response_json = await response.json()
                                response_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                                
                                # 计算完成token
                                try:
                                    completion_tokens = len(self.encoding.encode(response_text))
                                    total_tokens = len(self.encoding.encode(prompt)) + completion_tokens
                                    logger.info(f"完成token数: {completion_tokens}, 总token数: {total_tokens}")
                                except Exception as e:
                                    logger.warning(f"Token计数失败: {e}")
                                
                                # 解析响应
                                result = self._parse_entities_from_response(response_text)
                                break
                            else:
                                error_body = await response.text()
                                logger.error(f"API调用失败，状态码: {response.status}, 响应: {error_body}")
                                
                                # 检查是否需要重试
                                if response.status == 429 or response.status >= 500:
                                    should_retry = True
                                else:
                                    should_retry = False
                                    
                                if should_retry and retry_count < max_retries:
                                    retry_count += 1
                                    wait_time = 2 ** retry_count  # 指数回退
                                    logger.info(f"将在 {wait_time} 秒后重试 (尝试 {retry_count}/{max_retries})")
                                    await asyncio.sleep(wait_time)
                                else:
                                    return []
                
                except asyncio.TimeoutError:
                    logger.error(f"API调用超时")
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = 2 ** retry_count
                        logger.info(f"将在 {wait_time} 秒后重试 (尝试 {retry_count}/{max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        return []
                
                except Exception as e:
                    logger.error(f"API调用异常: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = 2 ** retry_count
                        logger.info(f"将在 {wait_time} 秒后重试 (尝试 {retry_count}/{max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        return []
            
            if result:
                if self.config.get("processing.cache_enabled", True):
                    self.config.get("cache")[cache_key] = result
                return result
            else:
                return []
        
        except Exception as e:
            logger.error(f"API调用异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def _call_gemini(self, prompt: str) -> List[Dict[str, Any]]:
        """
        调用Google Gemini API
        
        Args:
            prompt: 提示文本
            
        Returns:
            解析后的实体列表
        """
        if not self.gemini_model:
            logger.error("Gemini client not initialized. Cannot call LLM.")
            return []
             
        response_text = ""
        try:
            # 使用Gemini API异步调用
            logger.info("Calling Gemini API...")
            response = await self.gemini_model.generate_content_async(prompt)
            
            # 检查安全评级或阻塞
            if not response.candidates:
                prompt_feedback = response.prompt_feedback
                block_reason = prompt_feedback.block_reason if prompt_feedback else "Unknown"
                logger.error(f"Gemini API call blocked. Reason: {block_reason}")
                # Log safety ratings for debugging
                if prompt_feedback and prompt_feedback.safety_ratings:
                    for rating in prompt_feedback.safety_ratings:
                        logger.error(f"  Category: {rating.category}, Probability: {rating.probability}")
                return []

            # 提取文本
            if hasattr(response, 'text'):
                response_text = response.text
            elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                # Handle potential multi-part response, concatenate text parts
                response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            else:
                logger.error(f"Could not extract text from Gemini response. Response: {response}")
                return []

        except Exception as e:
            logger.error(f"Gemini API 调用异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        
        # 解析响应
        return self._parse_entities_from_response(response_text)
    
    def _parse_entities_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        从模型响应中解析出实体
        
        Args:
            response_text: 模型响应文本
            
        Returns:
            解析出的实体列表
        """
        try:
            # 保存原始响应到日志（前2000字符）
            logger.debug(f"模型原始响应(前2000字符): {response_text[:2000]}...")
            
            # 提取JSON部分
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]") + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.warning(f"响应中未找到JSON: {response_text[:500]}...")
                return []
            
            json_str = response_text[start_idx:end_idx]
            logger.debug(f"提取的JSON字符串(前1000字符): {json_str[:1000]}...")
            
            try:
                entities = json.loads(json_str)
                logger.info(f"成功解析JSON，包含 {len(entities)} 个原始实体")
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {str(e)}, JSON字符串: {json_str[:500]}...")
                return []
            
            # 验证格式
            valid_entities = []
            filtered_count = 0
            for idx, entity in enumerate(entities):
                if not isinstance(entity, dict):
                    logger.debug(f"实体 #{idx} 不是字典类型，被过滤")
                    filtered_count += 1
                    continue
                    
                if "entity_name" not in entity or "entity_type" not in entity or "entity_description" not in entity:
                    logger.debug(f"实体 #{idx} 缺少必要字段，被过滤: {entity}")
                    filtered_count += 1
                    continue
                
                if not entity["entity_name"] or not entity["entity_type"] or not entity["entity_description"]:
                    logger.debug(f"实体 #{idx} 包含空值字段，被过滤: {entity}")
                    filtered_count += 1
                    continue
                    
                valid_entities.append(entity)
            
            logger.info(f"验证后保留 {len(valid_entities)} 个有效实体，过滤掉 {filtered_count} 个无效实体")
            return valid_entities
            
        except Exception as e:
            logger.error(f"解析实体失败: {str(e)}\n响应: {response_text[:500]}...")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _post_process_entities(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """
        对提取的实体进行后处理
        
        Args:
            entities: 提取的实体列表
            text: 原始文本
            
        Returns:
            处理后的实体列表
        """
        # 标准化处理
        for entity in entities:
            # 确保entity_aliases是列表
            if "entity_aliases" not in entity or not isinstance(entity["entity_aliases"], list):
                entity["entity_aliases"] = []
            
            # 去除实体名称中的空格
            entity["entity_name"] = entity["entity_name"].strip()
        
        # 实体去重
        unique_entities = {}
        for entity in entities:
            name = entity["entity_name"]
            
            if name in unique_entities:
                # 合并描述
                existing_description = unique_entities[name]["entity_description"]
                new_description = entity["entity_description"]
                
                # 防止重复内容
                if new_description not in existing_description:
                    combined_description = f"{existing_description}\n\n{new_description}"
                    unique_entities[name]["entity_description"] = combined_description
                
                # 合并别名
                aliases = set(unique_entities[name]["entity_aliases"])
                aliases.update(entity["entity_aliases"])
                unique_entities[name]["entity_aliases"] = list(aliases)
            else:
                unique_entities[name] = entity
        
        return list(unique_entities.values())

async def extract_entities_from_directory(input_dir: str, output_dir: str, config_path: Optional[str] = None, target_filename: Optional[str] = None):
    """
    从目录中提取实体
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        config_path: 配置文件路径
        target_filename: 目标文件名，如果提供，则只处理该文件
    """
    extractor = EntityExtractor(config_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有文件
    files = []
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            # 恢复原来的逻辑：处理所有非隐藏文件
            if not filename.startswith('.'):
                files.append(os.path.join(root, filename))
    
    logger.info(f"找到 {len(files)} 个文件") # 恢复原来的日志信息
    
    # 批处理参数
    batch_size = extractor.config.get("processing.batch_size", 10)
    
    # 按批次处理
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        tasks = []
        
        for file_path in batch_files:
            file_id = os.path.basename(file_path)
            
            # 添加判断：只处理目标文件
            if target_filename and file_id != target_filename:
                logger.info(f"跳过非目标文件: {file_id}")
                continue
                
            output_path = os.path.join(output_dir, f"{file_id}_entities.json")
            
            # 如果已经处理过且启用缓存，则跳过
            if os.path.exists(output_path) and extractor.config.get("processing.cache_enabled", True):
                logger.info(f"跳过已处理文件: {file_id}")
                continue
                
            # 读取文件内容
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 创建提取任务
                task = asyncio.create_task(process_file(extractor, content, file_id, output_path))
                tasks.append(task)
                
            except Exception as e:
                logger.error(f"读取文件 {file_path} 失败: {str(e)}")
        
        # 等待当前批次完成
        if tasks:
            await asyncio.gather(*tasks)
            logger.info(f"完成批次 {i//batch_size + 1}/{(len(files) + batch_size - 1)//batch_size}")
    
    logger.info("实体提取完成")


async def process_file(extractor: EntityExtractor, content: str, file_id: str, output_path: str):
    """
    处理单个文件
    
    Args:
        extractor: 实体提取器
        content: 文件内容
        file_id: 文件ID
        output_path: 输出路径
    """
    try:
        logger.info(f"处理文件: {file_id}")
        entities = await extractor.extract_entities(content, file_id)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        
        logger.info(f"文件 {file_id} 实体提取完成，共 {len(entities)} 个实体")
        
    except Exception as e:
        logger.error(f"处理文件 {file_id} 失败: {str(e)}") 