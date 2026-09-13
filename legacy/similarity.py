import pandas as pd
import numpy as np

# 1. 读数据
book = pd.read_parquet(r'D:\pycharm project\Lib3\DataLoad\user_vec.parquet')

# 2. 建立 id -> vector 的映射
id2vec = dict(zip(book['user_id'], book['user_vector']))

def cosine_similarity_by_id(id1: int, id2: int) -> float:
    """
    根据两个 book_id 计算余弦相似度
    :return: 余弦相似度 [-1, 1]
    :raises: KeyError 如果 id 不存在
    """
    v1 = id2vec[id1]
    v2 = id2vec[id2]
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

# ------------------- 测试 -------------------
if __name__ == '__main__':
    print(cosine_similarity_by_id(1448, 5))