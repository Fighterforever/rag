#!/bin/bash

# 比较三个不同模型 (Qwen, Gemini, DeepSeek) 在同一文档上的结果差异
# 该脚本启用DEBUG日志级别，捕获详细的处理过程

# 设置工作目录
cd /Users/rach/RAG/telecom_kg

# 设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 源文件路径
SOURCE_FILE="_中国移动通信集团四川有限公司传输网络资源管理细则_chunks.json"
INPUT_DIR="/Users/rach/RAG/output"  # 确保这是正确的输入路径

# 输出目录
DEBUG_DIR="output/debug_comparison"
mkdir -p $DEBUG_DIR/deepseek
mkdir -p $DEBUG_DIR/logs

# 备份原始配置文件
cp config/settings.yaml config/settings.yaml.bak

# 启用调试日志
sed -i.logging 's/level: "INFO"/level: "DEBUG"/' config/settings.yaml

echo "====================================================="
echo "  开始调试deepseekr1"
echo "  源文件: $SOURCE_FILE"
echo "  源文件路径: $INPUT_DIR/$SOURCE_FILE"
echo "====================================================="

# 检查源文件是否存在
if [ ! -f "$INPUT_DIR/$SOURCE_FILE" ]; then
    echo "错误: 源文件不存在: $INPUT_DIR/$SOURCE_FILE"
    echo "请确认文件路径是否正确"
    exit 1
fi

# 运行DeepSeek模型
echo "运行DeepSeek模型..."
sed -i.model 's/active_model: ".*"/active_model: "default_chat_model"/' config/settings.yaml

echo "DeepSeek模型 - 提取实体..."
python src/main.py extract_entities \
  --input "$INPUT_DIR" \
  --output "$DEBUG_DIR/deepseek/entities" \
  --target_file "$SOURCE_FILE" \
  > $DEBUG_DIR/logs/deepseek_entity.log 2>&1

echo "DeepSeek模型 - 提取关系..."
python src/main.py extract_relationships \
  --entities "$DEBUG_DIR/deepseek/entities" \
  --input "$INPUT_DIR" \
  --output "$DEBUG_DIR/deepseek/relationships" \
  --target_file "$SOURCE_FILE" \
  > $DEBUG_DIR/logs/deepseek_relation.log 2>&1

# 恢复原始配置文件
mv config/settings.yaml.bak config/settings.yaml

# 分析结果
echo "====================================================="
echo "  结果分析"
echo "====================================================="
