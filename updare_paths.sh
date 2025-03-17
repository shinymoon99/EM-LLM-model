#!/bin/bash

# 定义旧路径和新路径
OLD_PATH="/home/chenrongyi/miniconda3/bin/python"
NEW_PATH="/datanfs2/chenrongyi/miniconda3/bin/python"
NEW_PATH_PYTHON3_12="/datanfs2/chenrongyi/miniconda3/bin/python3.12"

# 定义需要修改的文件列表
FILES=(
    "/datanfs2/chenrongyi/miniconda3/bin/jsonpatch"
    "/datanfs2/chenrongyi/miniconda3/bin/huggingface-cli"
    "/datanfs2/chenrongyi/miniconda3/bin/jsonpointer"
    "/datanfs2/chenrongyi/miniconda3/bin/pip3"
    "/datanfs2/chenrongyi/miniconda3/bin/conda"
    "/datanfs2/chenrongyi/miniconda3/bin/archspec"
    "/datanfs2/chenrongyi/miniconda3/bin/cph"
    "/datanfs2/chenrongyi/miniconda3/bin/pip"
    "/datanfs2/chenrongyi/miniconda3/bin/pydoc3.12"
    "/datanfs2/chenrongyi/miniconda3/bin/distro"
    "/datanfs2/chenrongyi/miniconda3/bin/idle3.12"
    "/datanfs2/chenrongyi/miniconda3/bin/jsondiff"
    "/datanfs2/chenrongyi/miniconda3/bin/conda-env"
    "/datanfs2/chenrongyi/miniconda3/bin/tqdm"
    "/datanfs2/chenrongyi/miniconda3/bin/normalizer"
    "/datanfs2/chenrongyi/miniconda3/bin/conda-content-trust"
    "/datanfs2/chenrongyi/miniconda3/bin/2to3-3.12"
    "/datanfs2/chenrongyi/miniconda3/bin/wheel"
)

# 遍历文件列表并修改
for FILE in "${FILES[@]}"; do
    if [[ -f "$FILE" ]]; then
        # 检查文件的第一行是否是旧路径
        if grep -q "^#!${OLD_PATH}$" "$FILE"; then
            # 如果是旧路径，替换为新路径
            sed -i "1s|^#!${OLD_PATH}$|#!${NEW_PATH}|" "$FILE"
            echo "已修改文件: $FILE"
        elif grep -q "^#!${OLD_PATH}3.12$" "$FILE"; then
            # 如果是旧路径的 Python 3.12，替换为新路径的 Python 3.12
            sed -i "1s|^#!${OLD_PATH}3.12$|#!${NEW_PATH_PYTHON3_12}|" "$FILE"
            echo "已修改文件: $FILE"
        else
            echo "无需修改文件: $FILE"
        fi
    else
        echo "文件不存在: $FILE"
    fi
done

echo "所有文件处理完成！"