#!/bin/bash

# 通信领域知识图谱构建工具运行脚本

# 设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 创建必要的目录
mkdir -p config data/cache output/entities output/relationships

# 检查settings.yaml是否存在
if [ ! -f "config/settings.yaml" ]; then
    echo "错误: 配置文件 config/settings.yaml 不存在，请先复制 settings.yaml 到 config 目录"
    exit 1
fi

# 检查命令行参数
if [ $# -eq 0 ]; then
    echo "使用方法: $0 [命令] [参数]"
    echo "可用命令:"
    echo "  extract_entities     - 提取实体"
    echo "  extract_relationships - 提取关系"
    echo "  store_kg             - 存储知识图谱"
    echo "  full_process         - 执行完整流程"
    exit 1
fi

# 执行Python命令
CMD="python src/main.py $@"
echo "执行命令: $CMD"
eval $CMD 