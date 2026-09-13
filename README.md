# Book Recommendation System — AIC 2025

基于矩阵分解（MF）和内容增强矩阵分解（CBMF）的图书推荐系统。

## 项目结构

```
├── src/                        # 核心源码
│   ├── config/                 #   配置 (BaseConfig → MFConfig / CBMFConfig)
│   ├── data/                   #   数据加载、预处理、数据集划分
│   ├── models/                 #   模型定义 (MF / CBMF)
│   ├── training/               #   训练器 (BaseTrainer → MFTrainer / CBMFTrainer)
│   ├── evaluation/             #   推理、F1评估、随机搜索优化
│   └── utils/                  #   日志、IO、向量加载、相似度
├── cli/                        # 命令行入口
│   ├── preprocess.py           #   数据预处理
│   ├── build_matrix.py         #   构建交互矩阵
│   ├── train.py                #   训练 (--model mf|cbmf)
│   ├── inference.py            #   推理生成推荐
│   └── evaluate.py             #   评估 F1 分数
├── DataLoad/                   # 原始数据
├── MyModule/                   # SentenceTransformer 模型文件
├── models/                     # 训练好的 PyTorch 模型
├── submissions/                # 推荐结果 CSV 输出
├── legacy/                     # 重构前的旧代码（归档）
├── pyproject.toml              # 项目元数据
├── config.yaml                 # 运行时配置（可选）
└── README.md
```

## 快速开始

```bash
# 1. 安装依赖
pip install torch numpy pandas scikit-learn matplotlib tqdm pyarrow

# 2. 构建矩阵
python cli/build_matrix.py --max-users 1000

# 3. 训练（或跳过，用已有模型）
python cli/train.py --model mf --epochs 100

# 4. 推理
python cli/inference.py --model mf --top-k 1

# 5. 评估
python cli/evaluate.py submissions/submission.csv
```

## 配置

环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BOOKREC_DATA_DIR` | `./DataLoad` | 数据目录 |
| `BOOKREC_MODEL_DIR` | `./MyModule` | 模型目录 |
| `BOOKREC_OUTPUT_DIR` | `./models` | 模型保存目录 |
| `BOOKREC_LOG_DIR` | `./logs` | 日志目录 |

## 重构改进

1. **分层包结构** — config / data / models / training / evaluation / utils
2. **消除代码重复** — 共享逻辑抽取到基类，不再有重复函数
3. **Bug 修复** — ID映射全量生成、移除死代码、修复路径硬编码
4. **CLI 工具** — 统一命令行入口替代散落的 `__main__` 块
5. **安全加载** — 模型加载使用 `weights_only=True`（兼容旧格式）
