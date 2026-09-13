# Book Recommendation System — AIC 2025

基于矩阵分解（MF）和内容增强矩阵分解（CBMF）的图书推荐系统。

---

## ⚠️ 重要更正

**本方案此前上报的 F1 = 0.0376 不是模型效果，是自评泄漏。**

| | 上报值 | 诚实值 |
|---|---|---|
| 提交文件 | `submissions/submission.csv` | `submissions/submission_v2.csv` |
| 指标 | F1 0.0376（precision 恒为 1.0） | top-1 accuracy ≈ 0.0196–0.0225 |
| 评测方式 | 对**训练所用的同一个文件**自评 | 4 个时序 cut-off，训练严格早于预测 |

原因有三，详见 **[FIXES.md](FIXES.md)**：

1. 提交的 1000 条预测中有 997 条是该用户在 `inter_reevaluation.csv` 中
   **最后一次借阅的书**，而 `cli/evaluate.py` 又拿同一个文件当验证集；
2. `src/evaluation/f1.py` 只按 `user_id` 合并，导致**精确率恒为 1.0**、
   F1 退化为与"选了哪本书"无关的常数；
3. `src/data/dataset.py` 里所谓的"时序切分"实际是按书的列位置排序
   （书号在几行前已被覆盖成位置编号），并非按借阅时间。

修复后请使用：

```bash
python cli/validate.py --report reports/validation.md   # 诚实评测（4 个时序 cut-off）
python cli/submit.py --tau 60 --cf-weight 0.25          # 生成合法提交
python -m pytest tests -q                               # 10 个测试
```

**诚实的结论**：数据里有真实时间戳（2020-01-02 → 2024-12-30），
在被正确使用时，本任务的天花板约 **2%**；"预测用户最近借的书"这一朴素规则
已拿到 0.0196，与调优后的时间衰减画像模型（0.0225）在统计上无法区分
（合并 4 个 cut-off 的配对比较：+16 人 / −11 人，共 1797 人）。

---

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
