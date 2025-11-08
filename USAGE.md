# 通用项目结构指南

本项目采用标准化目录结构，便于**代码、配置、输入、输出与文档**的统一管理，适用于脚本工具、数据分析、Web 项目、实验原型等各类个人项目。

## 📂 目录结构说明

- **input/**：原始输入文件（只读，不修改）
- **output/**：处理结果、生成文件、中间产物
- **assets/**：静态资源（图表、图片、音视频等）
  - `temp/`：临时调试资源（可随时清空）
- **src/**：核心源代码
- **scripts/**：可执行脚本（每个脚本完成单一任务）
- **configs/**：配置文件（YAML/JSON/TOML 等）
- **docs/**：项目文档、说明、设计稿
- **notebooks/**：交互式探索环境（Jupyter/Pluto 等）

## 🌐 环境变量加载

项目根目录的 `.env` 文件定义了关键路径，推荐使用对应语言的 dotenv 库加载：

### Python

```python
from dotenv import load_dotenv
import os
load_dotenv()
output_dir = os.environ["OUTPUT_DIR"]
```

### Rust

```rust
use dotenvy::dotenv;
dotenv().ok();
let output_dir = std::env::var("OUTPUT_DIR").unwrap();
```

### Julia / Bash / 其他

详见各语言 dotenv 文档。

## 🔄 推荐工作流

1. 将原始文件放入 `input/`
2. 编写脚本 → 放入 `scripts/`
3. 脚本读取 `input/`，输出到 `output/` 或 `assets/`
4. 所有路径通过 `.env` 获取，避免硬编码
5. 文档写入 `docs/`，便于回顾与分享

> ✅ 此结构支持**完全可复现**的工作流，提升个人效率与项目可维护性。
