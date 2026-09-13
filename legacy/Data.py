import pandas as pd, json, numpy as np
from pathlib import Path

dir_ = Path(r'DataLoad')

# 1. 读交互字典
inter2user = json.loads((dir_ / 'inter2user.json').read_text())
inter2book = json.loads((dir_ / 'inter2book.json').read_text())
inter2rating = json.loads((dir_ / 'inter2rating.json').read_text())

inter_df = pd.DataFrame({
 'user_id': pd.Series(inter2user, dtype='int'),
'book_id': pd.Series(inter2book, dtype='int'),
 'rating':  pd.Series(inter2rating, dtype='float')
})

# 2. 读 item 全集
item_df = pd.read_csv(dir_ / 'item.csv', encoding='ANSI')
all_books = item_df['book_id'].sort_values().unique()    # 5 w+

# 3. 骨架：所有用户 × 所有书，初值 NaN
users = inter_df['user_id'].unique()
skeleton = pd.DataFrame(index=pd.Index(users, name='user_id'),
                    columns=pd.Index(all_books, name='book_id'),
                   dtype='float')


g = inter_df.groupby(['user_id', 'book_id'])['rating'].max()
skeleton.update(g.unstack(level='book_id'))

inter_matrix = skeleton
inter_matrix.to_parquet(dir_ / 'user_inter_matrix.parquet')
print('形状：', inter_matrix.shape)
import json

 # 加载数据并创建映射
data = pd.read_parquet("DataLoad/user_inter_matrix.parquet")
user_ids = data.index.tolist()

 # 创建映射
index_to_user = {i: user_id for i, user_id in enumerate(user_ids[:1000])}
user_to_index = {user_id: i for i, user_id in enumerate(user_ids[:1000])}

print("映射完成！")
print(f"索引0对应的用户: {index_to_user[0]}")
print(f"用户{index_to_user[0]}对应的索引: {user_to_index[index_to_user[0]]}")

 # 保存为JSON
with open('id_mappings.json', 'w', encoding='utf-8') as f:
    json.dump({
        'index_to_user': index_to_user,
        'user_to_index': user_to_index
    }, f, ensure_ascii=False, indent=2)

print("✅ 映射已保存到 id_mappings.json")