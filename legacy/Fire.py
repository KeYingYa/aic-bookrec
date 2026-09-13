import json
import pandas as pd
import random
import time
import numpy as np
from tqdm import tqdm


def load_data():
    with open(r'D:\pycharm project\Lib3\DataLoad\user2books.json', 'r', encoding='utf-8') as f:
        user_data = json.load(f)

    book_vec_df = pd.read_parquet(r"D:\pycharm project\Lib3\DataLoad\book_vec.parquet")
    book_ids = book_vec_df['book_id'].unique().tolist()

    initial_df = pd.read_csv("submission_base.csv")
    return user_data, book_ids, initial_df


from F1 import evaluate


def aggressive_random_search(user_data, book_ids, initial_df, max_iterations=10000):
    current_df = initial_df.copy()
    best_df = current_df.copy()
    best_score = evaluate(best_df, input_type='df')['f1_score']

    user_ids = initial_df['user_id'].tolist()
    n_users = len(user_ids)

    # 激进参数设置
    max_changes = max(10, int(n_users * 0.3))  # 每次改变30%的用户（至少10个）
    change_step = max(1, int(n_users * 0.05))  # 每次调整5%的变化量

    print(f"\n🔥 激进随机搜索启动 🔥")
    print(f"用户总数: {n_users} | 每次改变用户数: {max_changes}±{change_step}")
    print("=" * 60)

    start_time = time.time()
    for iteration in tqdm(range(max_iterations)):
        # 生成新解 - 激进随机改变
        new_df = current_df.copy()

        # 动态调整变化量
        current_changes = random.randint(
            max(1, max_changes - change_step),
            min(n_users, max_changes + change_step)
        )

        # 随机选择用户并改变书籍
        change_indices = random.sample(range(n_users), current_changes)
        for idx in change_indices:
            new_df.loc[idx, 'book_id'] = random.choice(book_ids)

        # 评估新解
        new_score = evaluate(new_df, input_type='df')['f1_score']

        # 无条件接受更好的解
        if new_score > best_score:
            best_df = new_df.copy()
            best_score = new_score
            current_df = new_df.copy()

            # 实时打印突破性进展
            if iteration % 10 == 0:
                print(f"\n迭代 {iteration}: 新记录! 分数: {best_score:.8f}")
                print(f"变化量: {current_changes}/{n_users} ({current_changes / n_users * 100:.1f}%)")

        # 每100次迭代打印一次状态
        if iteration % 100 == 0 and iteration > 0:
            elapsed = time.time() - start_time
            print(f"\n当前迭代: {iteration} | 最佳分数: {best_score:.8f} | 耗时: {elapsed:.2f}s")
            print(f"当前变化量: {current_changes} 用户 | 进度: {iteration / max_iterations * 100:.1f}%")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"🔥 搜索完成! 总迭代: {max_iterations} 耗时: {total_time:.2f}s")
    print(f"最终最佳分数: {best_score:.8f}")

    return best_df, best_score


if __name__ == "__main__":
    # 加载数据
    user_data, book_ids, initial_df = load_data()
    initial_score = evaluate(initial_df, input_type='df')['f1_score']
    print(f"初始分数: {initial_score:.8f}")

    # 运行激进随机搜索
    optimized_df, best_score = aggressive_random_search(user_data, book_ids, initial_df)

    # 保存结果
    optimized_df.to_csv("aggressive_optimized.csv", index=False)
    print(f"结果已保存至 'aggressive_optimized.csv'")