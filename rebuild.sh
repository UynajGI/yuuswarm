#!/bin/bash

echo "🚀 Cleaning and Rebuilding in RELEASE mode..."

# 1. 如果 build 目录存在，为了保险起见，可以删掉缓存文件 (或者直接 rm -rf build 全删)
# 这里推荐比较温和的做法：保留目录，但重新 cmake
if [ ! -d "build" ]; then
  mkdir build
fi

cd build

# 2. 强制 Release 配置
cmake -DCMAKE_BUILD_TYPE=Release ..

# 3. 多核编译 (这里用了 -j4，防止把登录节点卡死)
make -j4

# 4. 检查编译结果
if [ $? -eq 0 ]; then
    echo "✅ Build Success! Executable is ready."
else
    echo "❌ Build Failed!"
    exit 1
fi