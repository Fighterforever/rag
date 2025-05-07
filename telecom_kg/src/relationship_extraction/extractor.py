"""
关系提取模块
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
import aiohttp
# import tiktoken  # 注释掉直接导入
import re

# 导入Google Gemini API库
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

class RelationshipExtractor:
    """关系提取器，用于从文本中提取实体间的关系"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化关系提取器
        
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
                # 从配置中获取Gemini API密钥和模型名称
                gemini_api_key = self.config.get("gemini.api_key")
                gemini_model_name = self.config.get("gemini.model_name", "gemini-2.5-pro-exp-03-25")
                
                if gemini_api_key:
                    genai.configure(api_key=gemini_api_key)
                    # 配置生成模型
                    generation_config = {
                        "temperature": 0.2,  # 低温度确保更确定性的输出
                        "top_p": 0.8,
                        "top_k": 40
                    }
                    
                    # 初始化Gemini模型
                    self.gemini_model = genai.GenerativeModel(
                        model_name=gemini_model_name,
                        generation_config=generation_config
                    )
                    logger.info(f"Google Gemini client initialized successfully with model: {gemini_model_name}")
                else:
                    logger.error("Gemini API key not found in configuration (gemini.api_key).")
                    raise ValueError("Missing Gemini API Key")
            except Exception as e:
                logger.error(f"Failed to initialize Google Gemini client: {e}")
        else:
            # 使用OpenAI兼容API (默认模型或Qwen)
            self.model_config = self.config.get_model_config(self.active_model_id)
            logger.info(f"使用OpenAI兼容API: {self.model_config.get('model')} ({self.model_config.get('api_base')})")
            # 创建token计数器 (用于估算请求大小)
            try:
                encoding_name = self.model_config.get("encoding_model", "cl100k_base")
                self.encoding = get_encoding(encoding_name)
                logger.info(f"成功初始化token计数器: {encoding_name}")
            except Exception as e:
                logger.warning(f"初始化token计数器失败: {e}，使用简单字符计数")
                self.encoding = SimpleTikToken(self.model_config.get("encoding_model", "cl100k_base"))
        
        # 加载提示模板
        prompt_path = os.path.join(self.prompts_dir, self.config.get("relationship_extraction.prompt_template"))
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
    
    async def extract_relationships(self, text: str, entities: List[Dict[str, Any]], text_id: str) -> List[Dict[str, Any]]:
        """
        从文本中提取实体间的关系
        
        Args:
            text: 输入文本
            entities: 已提取的实体列表
            text_id: 文本ID，用于缓存
            
        Returns:
            提取出的关系列表
        """
        # 检查缓存
        cache_file = os.path.join(self.cache_dir, f"relationship_{text_id}.json")
        if os.path.exists(cache_file) and self.config.get("processing.cache_enabled", True):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    logger.info(f"从缓存加载关系: {text_id}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载缓存失败: {str(e)}")
        
        # 如果实体太少，不提取关系
        if len(entities) < 2:
            logger.info(f"实体数量太少，不提取关系: {len(entities)}")
            return []
        
        # 构造提示
        entities_json = json.dumps(entities, ensure_ascii=False, indent=2)
        prompt = self.prompt_template.replace("{input_text}", text).replace("{entities_json}", entities_json)
        
        # 调用模型
        relationships = await self._call_llm(prompt)
        
        # 后处理
        relationships = self._post_process_relationships(relationships, entities)
        
        # 保存缓存
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(relationships, f, ensure_ascii=False, indent=2)
        
        return relationships
    
    async def _call_llm(self, prompt: str) -> List[Dict[str, Any]]:
        """
        调用大模型提取关系
        
        Args:
            prompt: 提示文本
            
        Returns:
            LLM返回的关系列表
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
            解析后的关系列表
        """
            # 计算token数
        try:
            prompt_tokens = len(self.encoding.encode(prompt))
            logger.info(f"提示token数: {prompt_tokens}")
        except Exception as e:
            logger.warning(f"Token计数失败: {e}")
        
        # 创建缓存键
        import hashlib
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        cache_key = f"{self.active_model_id}_{prompt_hash}"
        cache_file = os.path.join(self.cache_dir, f"llm_response_{cache_key}.json")
        
        # 检查缓存
        if os.path.exists(cache_file) and self.config.get("processing.cache_enabled", True):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    logger.info(f"从缓存加载LLM响应: {cache_key}")
                    cached_data = json.load(f)
                    return self._parse_relationships_from_response(cached_data["response"])
            except Exception as e:
                logger.warning(f"加载缓存失败: {str(e)}")
        
        # 准备API参数
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.model_config.get('api_key')}"
        }

        # 根据不同的模型调整max_tokens
        max_tokens = self.model_config.get("max_tokens", 8192)
        model_id = self.model_config.get("model", "").lower()
        
        # 对不同模型设置不同的max_tokens上限
        if "qwen" in model_id:
            if max_tokens > 8192:
                logger.warning(f"Qwen模型max_tokens上限为8192，当前值({max_tokens})已超过上限，将调整为8192")
                max_tokens = 8192
        elif "chatglm" in model_id:
            if max_tokens > 6144:
                logger.warning(f"ChatGLM模型max_tokens上限为6144，当前值({max_tokens})已超过上限，将调整为6144")
                max_tokens = 6144
        elif "baichuan" in model_id:
            if max_tokens > 4096:
                logger.warning(f"Baichuan模型max_tokens上限为4096，当前值({max_tokens})已超过上限，将调整为4096")
                max_tokens = 4096

        payload = {
            "model": self.model_config.get("model"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "top_p": 0.8,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }

        api_url = f"{self.model_config.get('api_base')}/chat/completions"
        max_retries = self.model_config.get("max_retries", 5)
        retry_count = 0
        response_text = ""

        # 执行API调用，带重试
        while retry_count <= max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    logger.info(f"正在调用API: {api_url}, 模型: {model_id}, max_tokens: {max_tokens}")
                    async with session.post(api_url, headers=headers, json=payload, timeout=300) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            response_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                            
                            # 计算完成token
                            try:
                                completion_tokens = len(self.encoding.encode(response_text))
                                total_tokens = prompt_tokens + completion_tokens
                                logger.info(f"完成token数: {completion_tokens}, 总token数: {total_tokens}")
                            except Exception as e:
                                logger.warning(f"Token计数失败: {e}")
                            
                            logger.info(f"原始 LLM 响应: {response_text[:500]}...") # 只记录部分响应避免日志过长
                            
                            # 保存缓存
                            try:
                                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                                with open(cache_file, 'w', encoding='utf-8') as f:
                                    json.dump({"response": response_text}, f, ensure_ascii=False)
                                    logger.info(f"LLM响应已缓存: {cache_key}")
                            except Exception as e:
                                logger.warning(f"保存缓存失败: {str(e)}")
                            
                            relationships = self._parse_relationships_from_response(response_text)
                            
                            return relationships
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
        
                        
                    
    async def _call_gemini(self, prompt: str) -> List[Dict[str, Any]]:
        """
        调用Google Gemini API提取关系
        
        Args:
            prompt: 提示文本
            
        Returns:
            LLM返回的关系列表
        """
        if not self.gemini_model:
            logger.error("Gemini client not initialized. Cannot call LLM.")
            return []
        
        response_text = ""
        max_retries = self.config.get("gemini.max_retries", 3)
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # 使用Gemini API异步调用
                logger.info("Calling Gemini API...")
                response = await self.gemini_model.generate_content_async(prompt)
                
                # 检查安全评级或阻塞
                if not response.candidates:
                    prompt_feedback = response.prompt_feedback
                    block_reason = prompt_feedback.block_reason if prompt_feedback else "Unknown"
                    logger.error(f"Gemini API call blocked. Reason: {block_reason}")
                    # 记录安全评级以进行调试
                    if prompt_feedback and prompt_feedback.safety_ratings:
                        for rating in prompt_feedback.safety_ratings:
                            logger.error(f"  Category: {rating.category}, Probability: {rating.probability}")
                    return []
                
                # 提取文本
                if hasattr(response, 'text'):
                    response_text = response.text
                elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    # 处理多部分响应，连接文本部分
                    response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                else:
                    logger.error(f"Could not extract text from Gemini response. Response: {response}")
                    return []
                
                logger.info(f"原始 LLM 响应: {response_text[:500]}...") # 只记录部分响应避免日志过长
                relationships = self._parse_relationships_from_response(response_text)
                
                return relationships
                
            except Exception as e:
                error_detail = str(e)
                # 获取详细的异常堆栈
                import traceback
                stack_trace = traceback.format_exc()
                
                logger.error(f"Gemini API调用异常: {error_detail}\n{stack_trace}")
                
                # 决定是否重试
                retry_count += 1
                if retry_count <= max_retries:
                    wait_time = 2 ** retry_count
                    logger.info(f"将在 {wait_time} 秒后重试 (尝试 {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"达到最大重试次数 ({max_retries})，放弃此次调用")
                    return []
        
        return []
    
    def _parse_relationships_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        从模型响应中解析出关系 (鲁棒版本，尝试处理截断)
        
        Args:
            response_text: 模型响应文本
            
        Returns:
            解析出的关系列表
        """
        logger.info(f"原始 LLM 响应 (用于解析): {response_text[:500]}...") # 只记录部分响应避免日志过长
        relationships = []
        json_str = None

        try:
            # 1. 尝试匹配 ```json ... ```
            match = re.search(r"```json\s*(\[.*?\])\s*```", response_text, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    relationships = json.loads(json_str)
                    logger.info(f"通过 ```json 标记成功解析 {len(relationships)} 个关系")
                except json.JSONDecodeError as e:
                    logger.warning(f"解析 ```json 内部内容失败: {e}. 响应片段: {json_str[:200]}...")
                    json_str = None # 解析失败，尝试后备方法

            # 2. 如果还没有结果，尝试查找 [ 和 ] 包裹的JSON
            if not relationships:
                # 使用更严格的正则模式查找JSON数组
                match = re.search(r"\[\s*{[^[\]]*}\s*(?:,\s*{[^[\]]*}\s*)*\]", response_text, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    try:
                        relationships = json.loads(json_str)
                        logger.info(f"通过正则匹配JSON数组成功解析 {len(relationships)} 个关系")
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析正则匹配内容失败: {e}. 响应片段: {json_str[:200]}...")
                        json_str = None

            # 3. 如果前两种方法都失败，尝试查找最大的有效 JSON 列表
            if not relationships:
                start_idx = response_text.find("[")
                if start_idx != -1:
                    # 从开始位置向后查找，逐步增加范围尝试解析
                    for end_idx in range(start_idx + 2, len(response_text) + 1):  # +2确保至少包含一个字符
                        if end_idx >= len(response_text):
                            # 如果到达文本末尾，尝试解析整个剩余部分
                            potential_json_str = response_text[start_idx:]
                        else:
                            potential_json_str = response_text[start_idx:end_idx]
                        
                        # 确保JSON字符串是以 ] 结尾的
                        if not potential_json_str.rstrip().endswith(']'):
                            # 如果不是以 ] 结尾，且接近文本末尾，尝试添加 ]
                            if end_idx >= len(response_text) - 10 and potential_json_str.count('[') > potential_json_str.count(']'):
                                potential_json_str += ']' * (potential_json_str.count('[') - potential_json_str.count(']'))

                        try:
                            # 尝试解析，确保是列表
                            parsed_data = json.loads(potential_json_str)
                            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                                relationships = parsed_data
                                json_str = potential_json_str
                                logger.info(f"通过增量扫描成功解析 {len(relationships)} 个关系，长度: {len(json_str)}")
                                break
                        except json.JSONDecodeError:
                            # 这个长度解析失败，继续尝试
                            continue
                    
                    # 如果增量扫描失败，尝试从末尾向前扫描
                    if not relationships:
                        for end_idx in range(len(response_text), start_idx, -1):
                            potential_json_str = response_text[start_idx:end_idx]
                            
                            # 尝试修复不完整的JSON
                            if not potential_json_str.rstrip().endswith(']'):
                                if potential_json_str.count('[') > potential_json_str.count(']'):
                                    potential_json_str += ']' * (potential_json_str.count('[') - potential_json_str.count(']'))

                            try:
                                parsed_data = json.loads(potential_json_str)
                                if isinstance(parsed_data, list) and len(parsed_data) > 0:
                                    relationships = parsed_data
                                    json_str = potential_json_str
                                    logger.info(f"通过逆向扫描成功解析 {len(relationships)} 个关系。结束索引: {end_idx}")
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    if not relationships:
                        logger.warning(f"响应中未能解析出有效的JSON列表。尝试的范围: {start_idx} 到末尾")
                else:
                    logger.warning(f"响应中未找到 '[' 字符。")

            # 4. 如果仍未找到关系，尝试更加宽松的解析方式：逐个寻找对象
            if not relationships and '{' in response_text and '}' in response_text:
                try:
                    # 使用正则表达式找出所有可能的JSON对象
                    object_pattern = r'{[^{}]*(?:{[^{}]*}[^{}]*)*}'
                    potential_objects = re.findall(object_pattern, response_text)
                    
                    logger.debug(f"通过对象模式找到 {len(potential_objects)} 个潜在JSON对象")
                    found_valid_objects = 0
                    
                    for obj_str in potential_objects:
                        try:
                            obj = json.loads(obj_str)
                            # 检查是否包含必要的关系字段
                            if (isinstance(obj, dict) and 
                                "source_entity" in obj and 
                                "target_entity" in obj and 
                                "relation_type" in obj):
                                relationships.append(obj)
                                found_valid_objects += 1
                        except json.JSONDecodeError:
                            continue
                    
                    if relationships:
                        logger.info(f"通过逐个对象解析成功提取 {found_valid_objects} 个关系")
                except Exception as e:
                    logger.warning(f"逐个对象解析失败: {str(e)}")

            if not relationships:
                logger.warning(f"最终未能从响应中提取到有效的JSON列表。")
                return []
            
            # --- 后续验证和处理逻辑 ---
            valid_relationships = []
            filtered_count = 0
            
            for idx, relation in enumerate(relationships):
                if not isinstance(relation, dict):
                    logger.debug(f"关系 #{idx} 不是字典类型，被过滤")
                    filtered_count += 1
                    continue
                    
                required_fields = ["source_entity", "target_entity", "relation_type", "relation_description"]
                if not all(field in relation for field in required_fields):
                    missing_fields = [field for field in required_fields if field not in relation]
                    logger.debug(f"关系 #{idx} 缺少必要字段 {missing_fields}，被过滤: {relation}")
                    filtered_count += 1
                    continue
                
                if not all(relation[field] for field in required_fields):
                    empty_fields = [field for field in required_fields if not relation[field]]
                    logger.debug(f"关系 #{idx} 包含空值字段 {empty_fields}，被过滤: {relation}")
                    filtered_count += 1
                    continue
                    
                # 确保关系强度是整数
                try:
                    relation_strength = relation.get("relation_strength")
                    if relation_strength is None:
                        relation["relation_strength"] = 5
                        logger.debug(f"关系 #{idx} 未提供强度值，设置默认值 5")
                    else:
                        relation["relation_strength"] = int(relation_strength)
                        if relation["relation_strength"] < 1:
                            logger.debug(f"关系 #{idx} 强度值 {relation_strength} 小于1，设置为1")
                            relation["relation_strength"] = 1
                        elif relation["relation_strength"] > 10:
                            logger.debug(f"关系 #{idx} 强度值 {relation_strength} 大于10，设置为10")
                            relation["relation_strength"] = 10
                except (ValueError, TypeError):
                    logger.debug(f"关系 #{idx} 强度值 {relation.get('relation_strength')} 非法，设置默认值 5")
                    relation["relation_strength"] = 5
                    
                valid_relationships.append(relation)
            
            logger.info(f"验证后得到 {len(valid_relationships)} 个有效关系，过滤掉 {filtered_count} 个无效关系")
            return valid_relationships
            
        except Exception as e:
            # 捕获任何意外错误
            error_msg = f"解析关系时发生未知错误: {str(e)}"
            if json_str:
                error_msg += f"\n尝试解析的JSON字符串 (前200字符): {json_str[:200]}..."
            else:
                error_msg += f"\n未能定位JSON字符串。原始响应 (前500字符): {response_text[:500]}..."
            logger.error(error_msg, exc_info=True) # 添加 exc_info=True 获取堆栈跟踪
            return []
    
    def _post_process_relationships(self, relationships: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对提取的关系进行后处理
        
        Args:
            relationships: 提取的关系列表
            entities: 提取的实体列表
            
        Returns:
            处理后的关系列表
        """
        # 创建实体名称集合
        entity_names = {entity["entity_name"]: entity for entity in entities}
        logger.debug(f"实体列表包含 {len(entity_names)} 个实体")
        
        # 关系标准化和过滤
        valid_relationships = []
        unique_relations_check = set() # 用于更精细的去重
        filtered_count = 0
        filter_reasons = {
            "missing_field": 0,
            "entity_not_exist": 0,
            "self_loop": 0,
            "duplicate": 0
        }

        for idx, relation in enumerate(relationships):
            # 检查源实体和目标实体是否存在
            source_name = relation.get("source_entity")
            target_name = relation.get("target_entity")
            rel_type = relation.get("relation_type")

            if not source_name or not target_name or not rel_type:
                logger.debug(f"关系 #{idx} 缺少关键字段: {relation}")
                filtered_count += 1
                filter_reasons["missing_field"] += 1
                continue

            if source_name not in entity_names:
                logger.debug(f"关系 #{idx} 源实体 '{source_name}' 不存在于实体列表中")
                filtered_count += 1
                filter_reasons["entity_not_exist"] += 1
                continue
                
            if target_name not in entity_names:
                logger.debug(f"关系 #{idx} 目标实体 '{target_name}' 不存在于实体列表中")
                filtered_count += 1
                filter_reasons["entity_not_exist"] += 1
                continue
                
            # 确保不是自环关系
            if source_name == target_name:
                logger.debug(f"关系 #{idx} 是自环关系: {source_name}")
                filtered_count += 1
                filter_reasons["self_loop"] += 1
                continue
            
            # 添加实体类型信息 (如果实体存在)
            relation["source_entity_type"] = entity_names[source_name]["entity_type"]
            relation["target_entity_type"] = entity_names[target_name]["entity_type"]

            # 简单的唯一性检查 (源实体, 目标实体, 类型)
            # 简单的唯一性检查 (源实体, 目标实体, 类型)
            rel_key = (source_name, target_name, rel_type.lower())
            if rel_key not in unique_relations_check:
                valid_relationships.append(relation)
                unique_relations_check.add(rel_key)
            else:
                logger.debug(f"关系 #{idx} 是重复关系 (基于源-目标-类型): {rel_key}")
                filtered_count += 1
                filter_reasons["duplicate"] += 1


        # 关系去重与合并 (这个逻辑可以保留，或者用上面的简单去重替代)
        # 注意：如果上面已经做了基于 key 的去重，这里的合并逻辑可能只会处理描述或强度不同的情况
        merged_relationships = {}
        merge_count = 0
        
        for relation in valid_relationships: # 使用已经初步去重和验证的列表
            # 修改：将关系类型转换为小写进行比较
            relation_type_lower = relation["relation_type"].lower()
            key = (relation["source_entity"], relation["target_entity"], relation_type_lower)
            
            if key in merged_relationships:
                # 已存在，选择描述更详细或强度更高的关系
                existing = merged_relationships[key]
                merged = False
                
                # 合并策略：保留更长的描述
                if len(relation.get("relation_description", "")) > len(existing.get("relation_description", "")):
                    existing["relation_description"] = relation.get("relation_description", "")
                    merged = True
                    
                # 保留更高的强度
                if relation.get("relation_strength", 0) > existing.get("relation_strength", 0):
                    existing["relation_strength"] = relation.get("relation_strength", 0)
                    merged = True
                    
                if merged:
                    merge_count += 1
                    logger.debug(f"合并关系: {key}")
            else:
                # 确保所有需要的字段都存在，给默认值以防万一
                relation.setdefault("relation_description", "")
                relation.setdefault("relation_strength", 5) # 默认强度
                merged_relationships[key] = relation

        final_relationships = list(merged_relationships.values())
        
        logger.info(f"后处理完成，得到 {len(final_relationships)} 个关系。过滤掉 {filtered_count} 个关系 (原因: {filter_reasons})，合并了 {merge_count} 个关系。")
        return final_relationships

async def extract_relationships_from_directory(entity_dir: str, output_dir: str, input_dir: str, config_path: Optional[str] = None, target_filename: Optional[str] = None):
    """
    从目录中提取关系
    
    Args:
        entity_dir: 实体目录，包含已提取的实体文件
        output_dir: 输出目录
        input_dir: 原始文本目录
        config_path: 配置文件路径
        target_filename: 目标文件名，如果提供，则只处理该文件
    """
    extractor = RelationshipExtractor(config_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有实体文件
    entity_files = []
    for root, _, filenames in os.walk(entity_dir):
        for filename in filenames:
            if filename.endswith("_entities.json") and not filename.startswith('.'):
                entity_files.append(os.path.join(root, filename))
    
    logger.info(f"找到 {len(entity_files)} 个实体文件")
    
    # 批处理参数
    batch_size = extractor.config.get("processing.batch_size", 10)
    
    # 按批次处理
    for i in range(0, len(entity_files), batch_size):
        batch_files = entity_files[i:i+batch_size]
        tasks = []
        
        for entity_file in batch_files:
            # 基于实体文件名获取文本ID
            base_name = os.path.basename(entity_file)
            text_id = base_name.replace("_entities.json", "")
            
            # 如果指定了目标文件名，只处理该文件
            if target_filename and text_id != target_filename:
                logger.info(f"跳过非目标文件: {text_id}")
                continue
                
            output_path = os.path.join(output_dir, f"{text_id}_relationships.json")
            
            # 如果已经处理过且启用缓存，则跳过
            if os.path.exists(output_path) and extractor.config.get("processing.cache_enabled", True):
                logger.info(f"跳过已处理文件: {text_id}")
                continue
                
            # 查找原始文本文件
            text_file = find_text_file(input_dir, text_id)
            if not text_file:
                logger.error(f"未找到原始文本文件: {text_id}")
                continue
                
            # 创建处理任务
            task = asyncio.create_task(process_file(extractor, text_file, entity_file, output_path))
            tasks.append(task)
        
        # 等待当前批次完成
        if tasks:
            await asyncio.gather(*tasks)
            logger.info(f"完成批次 {i//batch_size + 1}/{(len(entity_files) + batch_size - 1)//batch_size}")
    
    logger.info("关系提取完成")


def find_text_file(input_dir: str, text_id: str) -> Optional[str]:
    """
    查找原始文本文件
    
    Args:
        input_dir: 原始文本目录
        text_id: 文本ID
        
    Returns:
        文本文件路径，未找到则返回None
    """
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename == text_id:
                return os.path.join(root, filename)
    return None


async def process_file(extractor: RelationshipExtractor, text_file: str, entity_file: str, output_path: str):
    """
    处理单个文件（使用块级并行处理）
    
    Args:
        extractor: 关系提取器
        text_file: 文本文件路径
        entity_file: 实体文件路径
        output_path: 输出路径
    """
    try:
        logger.info(f"处理文件: {os.path.basename(text_file)}")
        
        # 判断文件是否为chunks分块文件
        is_chunks_file = "_chunks.json" in text_file
        chunk_data = []
        
        if is_chunks_file:
            # 如果是chunks文件，读取其中的块列表
            with open(text_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            logger.info(f"检测到chunks格式文件，包含 {len(chunk_data)} 个文本块")
        else:
            # 如果是普通文本文件，直接读取内容
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()
                chunk_data = [{"content": content}]  # 创建单个块
        
        # 读取实体
        with open(entity_file, 'r', encoding='utf-8') as f:
            all_entities = json.load(f)
            
        # 按块并行处理
        if len(chunk_data) > 1:
            text_id = os.path.basename(text_file)
            chunk_limit = extractor.config.get("processing.chunk_batch_size", 5)  # 控制并发块数
            all_relationships = []
            
            # 分批处理块以控制并发量
            for i in range(0, len(chunk_data), chunk_limit):
                batch_chunks = chunk_data[i:i+chunk_limit]
                logger.info(f"处理批次 {i//chunk_limit + 1}/{(len(chunk_data) + chunk_limit - 1)//chunk_limit}, 共 {len(batch_chunks)} 个块")
                
                # 创建并行任务
                tasks = []
                for idx, chunk in enumerate(batch_chunks):
                    chunk_text = chunk["content"]
                    chunk_id = f"{text_id}_chunk_{i+idx+1}"
                    task = process_single_chunk(extractor, chunk_text, all_entities, chunk_id)
                    tasks.append(task)
                
                # 并行执行当前批次的任务
                chunk_results = await asyncio.gather(*tasks)
                
                # 合并结果
                for rels in chunk_results:
                    all_relationships.extend(rels)
                
                logger.info(f"处理批次 {i//chunk_limit + 1} 完成，已累计提取 {len(all_relationships)} 个原始关系")
            
            # 对合并后的所有关系进行后处理
            merged_relationships = merge_relationships(all_relationships, all_entities)
            logger.info(f"块级并行处理完成，合并后得到 {len(merged_relationships)} 个关系")
            
            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_relationships, f, ensure_ascii=False, indent=2)
            
            logger.info(f"文件 {text_id} 关系提取完成，共 {len(merged_relationships)} 个关系")
            
        else:
            # 单块处理（兼容旧逻辑）
            content = chunk_data[0]["content"]
            content = chunk_data[0]["content"]
            text_id = os.path.basename(text_file)
            relationships = await extractor.extract_relationships(content, all_entities, text_id)
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(relationships, f, ensure_ascii=False, indent=2)
        
        logger.info(f"文件 {text_id} 关系提取完成，共 {len(relationships)} 个关系")
        
    except Exception as e:
        logger.error(f"处理文件失败: {str(e)}", exc_info=True)


async def process_single_chunk(extractor: RelationshipExtractor, chunk_text: str, entities: List[Dict[str, Any]], chunk_id: str) -> List[Dict[str, Any]]:
    """
    处理单个文本块
    
    Args:
        extractor: 关系提取器
        chunk_text: 块文本内容
        entities: 实体列表
        chunk_id: 块ID
        
    Returns:
        块中提取的关系列表
    """
    try:
        logger.info(f"处理文本块: {chunk_id}")
        
        # 如果文本块太长，可能导致token超出限制，进行截断处理
        max_chunk_length = 10000  # 设置一个合理的最大长度
        if len(chunk_text) > max_chunk_length:
            logger.warning(f"文本块 {chunk_id} 过长，进行截断处理: {len(chunk_text)} -> {max_chunk_length}")
            chunk_text = chunk_text[:max_chunk_length]
        
        # 使用原有extract_relationships方法处理单个块
        relationships = await extractor.extract_relationships(chunk_text, entities, chunk_id)
        
        logger.info(f"文本块 {chunk_id} 提取完成，找到 {len(relationships)} 个关系")
        return relationships
        
    except Exception as e:
        logger.error(f"处理文本块失败: {chunk_id}, 错误: {str(e)}")
        return []


def merge_relationships(all_relationships: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    合并从多个块中提取的关系，去重并确保质量
    
    Args:
        all_relationships: 所有块中提取的关系列表
        entities: 实体列表
        
    Returns:
        合并后的关系列表
    """
    # 创建实体名称集合
    entity_names = {entity["entity_name"]: entity for entity in entities}
    
    # 关系标准化和过滤
    valid_relationships = []
    unique_relations_check = set()  # 用于去重
    
    for relation in all_relationships:
        # 检查源实体和目标实体是否存在
        source_name = relation.get("source_entity")
        target_name = relation.get("target_entity")
        rel_type = relation.get("relation_type")
        
        if not source_name or not target_name or not rel_type:
            continue
            
        if source_name not in entity_names or target_name not in entity_names:
            continue
            
        # 确保不是自环关系
        if source_name == target_name:
            continue
            
        # 添加实体类型信息
        relation["source_entity_type"] = entity_names[source_name]["entity_type"]
        relation["target_entity_type"] = entity_names[target_name]["entity_type"]
        
        # 简单的唯一性检查
        # 修改：将关系类型转换为小写进行比较
        rel_key = (source_name, target_name, rel_type.lower())
        if rel_key not in unique_relations_check:
            valid_relationships.append(relation)
            unique_relations_check.add(rel_key)
    
    # 进一步合并相同的关系，保留更详细的描述和更高的关系强度
    merged_relationships = {}
    for relation in valid_relationships:
        # 修改：将关系类型转换为小写进行比较
        relation_type_lower = relation["relation_type"].lower()
        key = (relation["source_entity"], relation["target_entity"], relation_type_lower)
        
        if key in merged_relationships:
            # 已存在关系，合并信息
            existing = merged_relationships[key]
            
            # 保留更长的描述
            if len(relation.get("relation_description", "")) > len(existing.get("relation_description", "")):
                existing["relation_description"] = relation.get("relation_description", "")
                
            # 保留更高的强度
            relation_strength = relation.get("relation_strength", 5)
            existing_strength = existing.get("relation_strength", 5)
            existing["relation_strength"] = max(relation_strength, existing_strength)
            
        else:
            # 新关系，确保有默认值
            relation.setdefault("relation_description", "")
            relation.setdefault("relation_strength", 5)
            merged_relationships[key] = relation
    
    result = list(merged_relationships.values())
    logger.info(f"块关系合并完成，从 {len(all_relationships)} 个原始关系中提取了 {len(result)} 个有效关系")
    return result 