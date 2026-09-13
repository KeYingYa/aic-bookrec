import json
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
def add_rating() -> pd.DataFrame:
    file_path = Path(r'DataLoad\inter_reevaluation.csv')
    df = pd.read_csv(file_path, encoding='ANSI')

    # ----- 你的时间清洗 + 评分逻辑（保持不动） -----
    borrow_time = pd.to_datetime(df['借阅时间'])
    return_time = pd.to_datetime(df['还书时间'], format='%Y-%m-%d%H:%M:%S')
    now_time = pd.Timestamp('now')
    actual_duration = (return_time - borrow_time).fillna(now_time - borrow_time)
    time = actual_duration.dt.total_seconds() / 86400

    dur = time.clip(lower=0, upper=365)
    dur_norm = (dur - dur.min()) / (dur.max() - dur.min() + 1e-8)
    reborrow = df['续借次数'].clip(upper=5)
    reborrow_norm = (reborrow - reborrow.min()) / (reborrow.max() - reborrow.min() + 1e-8)

    score = 0.8 * dur_norm + 0.2 * reborrow_norm
    df['rating'] = (score * 9 + 1).round(3)
    # ------------------------------------------

    # 构造三个一对一字典
    inter2user = dict(zip(df['inter_id'], df['user_id']))
    inter2book = dict(zip(df['inter_id'], df['book_id']))
    inter2rating = dict(zip(df['inter_id'], df['rating']))

    # 保存
    out_dir = Path(r'DataLoad')
    out_dir.mkdir(exist_ok=True)
    (out_dir / 'inter2user.json').write_text(json.dumps(inter2user, ensure_ascii=False, indent=2))
    (out_dir / 'inter2book.json').write_text(json.dumps(inter2book, ensure_ascii=False, indent=2))
    (out_dir / 'inter2rating.json').write_text(json.dumps(inter2rating, ensure_ascii=False, indent=2))

    return df


def tovec(csv_path=r'DataLoad\item.csv',
          model_dir='./MyModule',
          save_path=r'DataLoad\book_vec.parquet',
          id_col='book_id',
          batch_size=64):
    """从CSV读取数据，除指定列外全部拼接成句子，使用本地模型生成向量，保存结果到parquet文件"""
    try:
        from sentence_transformers import SentenceTransformer
        import pandas as pd
        from pathlib import Path
        import json
        import os

        # 路径处理
        csv_file, save_file = Path(csv_path), Path(save_path)

        # 修复模型目录结构（针对512维模型）
        model_path = Path(model_dir)
        pooling_dir = model_path / '1_Pooling'

        # 检查并创建必要的目录结构
        if not pooling_dir.exists():
            pooling_dir.mkdir(parents=True, exist_ok=True)

            # 创建pooling配置（使用512维）
            pooling_config = {
                "word_embedding_dimension": 512,
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": True,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False
            }

            with open(pooling_dir / 'config.json', 'w', encoding='utf-8') as f:
                json.dump(pooling_config, f, indent=2)

        # 检查modules.json文件
        modules_file = model_path / 'modules.json'
        if not modules_file.exists():
            modules_config = [
                {
                    "type": "sentence_transformers.models.Transformer",
                    "path": ".",
                    "tokenizer_name": "."
                },
                {
                    "type": "sentence_transformers.models.Pooling",
                    "word_embedding_dimension": 512,
                    "pooling_mode": "mean"
                }
            ]

            with open(modules_file, 'w', encoding='utf-8') as f:
                json.dump(modules_config, f, indent=2)

        # 加载模型
        model = SentenceTransformer(model_dir)

        # 读取CSV文件
        df = pd.read_csv(csv_file, encoding='ANSI')  # 确保编码方式与文件匹配

        # 指定不参与拼接的列
        text_cols = [c for c in df.columns if c != id_col]

        # 拼接文本列，用空格分隔，NaN自动变空字符串
        df['text'] = df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)

        # 生成向量
        embs = model.encode(df['text'].tolist(), normalize_embeddings=True, batch_size=batch_size)

        # 把numpy array存成一列
        df['book_vector'] = list(embs)

        # 只保留最终需要的列
        out_df = df[[id_col, 'book_vector']]

        # 保存到parquet文件
        save_file.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(save_file, index=False)

        print(f'✔ 已完成，结果写入 {save_file.absolute()}')
        print(f'生成的向量维度: {embs.shape[1]}')
        return out_df

    except Exception as e:
        print(f"处理过程中出现错误：{e}")
        return None

if __name__ == '__main__':
    add_rating()
    # tovec()
