# 通信领域知识图谱构建工具

一个基于大语言模型的通信领域知识图谱构建工具，专门针对通信行业专业文档进行优化。该工具基于Microsoft GraphRAG框架的思想，通过实体与关系提取、知识融合优化等步骤，将非结构化文本转换为结构化的知识图谱。

## 功能特点

- **领域专业化**：针对通信行业特有的实体类型和关系模型进行定制优化
- **智能提取**：使用大语言模型从文本中提取实体和关系
- **知识融合**：通过实体对齐和关系优化，消除冗余和矛盾
- **灵活配置**：通过YAML配置文件灵活调整各项参数
- **批量处理**：支持并行批量处理大量文档
- **图形存储**：使用Neo4j图数据库存储和查询知识图谱

## 系统要求

- Python 3.8+
- Neo4j 4.4+（用于知识图谱存储）
- 足够的磁盘空间用于缓存和处理大文件

## 安装方法

1. 克隆代码库
2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
3. 确保Neo4j数据库正在运行（如果需要存储功能）

## 配置说明

主要配置文件为`config/settings.yaml`，需要设置：

- **API密钥**：配置用于大语言模型访问的API密钥
- **输入/输出路径**：指定输入文档和输出结果的路径
- **Neo4j连接**：设置Neo4j数据库的连接信息
- **处理参数**：调整批处理大小、缓存策略等

## 使用方法

### 提取实体

```bash
python src/main.py extract_entities --input /path/to/input --output /path/to/output
```

### 提取关系

```bash
python src/main.py extract_relationships --entities /path/to/entities --input /path/to/input --output /path/to/output
```

### 存储知识图谱

```bash
python src/main.py store_kg --entities /path/to/entities --relationships /path/to/relationships
```

### 执行完整流程

```bash
python src/main.py full_process --input /path/to/input
```

## 目录结构

```
telecom_kg/
├── config/               # 配置文件
├── prompts/              # 提示模板
├── src/                  # 源代码
│   ├── entity_extraction/    # 实体提取模块
│   ├── relationship_extraction/ # 关系提取模块
│   ├── knowledge_fusion/    # 知识融合模块
│   ├── storage/          # 存储模块
│   ├── utils/            # 工具函数
│   └── main.py           # 主程序
├── data/                 # 数据目录
│   └── cache/            # 缓存目录
├── output/               # 输出目录
├── requirements.txt      # 依赖包列表
└── README.md             # 项目说明
```

## 优化与扩展

1. **提示模板优化**：可以根据特定领域需求调整`prompts/`目录下的提示模板
2. **实体类型扩展**：在配置文件中添加新的实体类型定义
3. **关系类型扩展**：在配置文件中定义新的关系类型
4. **批处理参数调优**：根据硬件性能调整批处理大小和并行度

## 示例应用

该工具可用于：
- 构建通信网络知识库
- 辅助技术文档检索与问答
- 支持专业知识学习与培训
- 辅助技术方案设计与评估 