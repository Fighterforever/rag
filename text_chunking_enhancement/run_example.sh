#!/bin/bash

# 文本分块系统示例运行脚本

# 确保脚本在错误时退出
set -e

# 设置默认路径
INPUT_DIR="../dataset"
OUTPUT_DIR="./output"
MAX_WORKERS=4

# 显示帮助信息
function show_help {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  -i, --input DIR      输入目录 (默认: $INPUT_DIR)"
    echo "  -o, --output DIR     输出目录 (默认: $OUTPUT_DIR)"
    echo "  -w, --workers NUM    并行处理数 (默认: $MAX_WORKERS)"
    echo "  -h, --help           显示帮助信息"
    echo "示例:"
    echo "  $0 -i ../dataset -o ./output -w 4"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    
    case $key in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -w|--workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 运行示例：单个文件处理
echo "正在处理单个示例文件..."
python main.py --input "$INPUT_DIR/专业规范_传输/_中国移动通信集团四川有限公司传输网通路组织管理细则（4.0）.md" \
               --output "$OUTPUT_DIR/single" \
               --extract-images \
               --max-chunk-size 1500 \
               --chunk-overlap 200

# 等待一下
sleep 2

# 运行示例：批量处理
echo "正在批量处理通信领域文档..."
python main.py --input "$INPUT_DIR/专业规范_传输" \
               --output "$OUTPUT_DIR/batch" \
               --batch \
               --extract-images \
               --max-chunk-size 1500 \
               --chunk-overlap 200 \
               --max-workers $MAX_WORKERS \
               --file-types .md .pdf

echo "处理完成。请查看 $OUTPUT_DIR 目录获取结果。" 