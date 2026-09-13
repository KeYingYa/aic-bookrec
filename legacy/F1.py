import pandas as pd
from typing import List, Tuple, Union

# ---------- 常量 ----------
VALID_PATH = r"D:\pycharm project\Lib3\DataLoad\inter_reevaluation.csv"

# ---------- 核心 F1 计算 ----------
def _f1(sub: pd.DataFrame, val: pd.DataFrame) -> Tuple[float, float, float]:
    """sub/val 都已对齐用户，且列名均为 ['user_id','book_id']"""
    if sub.empty or val.empty:
        return 0., 0., 0.

    # 内连接：只留双方同时出现的用户
    mrg = pd.merge(sub, val, on='user_id', suffixes=('_p', '_t'))
    tp = (mrg['book_id_p'] == mrg['book_id_t']).sum()
    fp = len(sub) - tp
    fn = len(val) - tp

    p = tp / (tp + fp) if (tp + fp) else 0.
    r = tp / (tp + fn) if (tp + fn) else 0.
    f1 = 2 * p * r / (p + r) if (p + r) else 0.
    return p, r, f1


# ---------- 统一对外接口 ----------
def evaluate(inp: Union[str, pd.DataFrame],
             *,
             user_col: str = 'user_id',
             book_col: str = 'book_id') -> dict:
    """
    整表/文件评估
    inp : csv 路径 或 DataFrame
    """
    try:
        sub = pd.read_csv(inp) if isinstance(inp, str) else inp.copy()
        sub = sub[[user_col, book_col]].rename(columns={user_col: 'user_id', book_col: 'book_id'})
    except Exception as e:
        return dict(precision=0, recall=0, f1_score=0, is_valid=False, message=f'读取提交失败: {e}')

    try:
        val = pd.read_csv(VALID_PATH, encoding='ANSI')\
                .rename(columns={user_col: 'user_id', book_col: 'book_id'})[['user_id', 'book_id']]\
                .drop_duplicates()
    except Exception as e:
        return dict(precision=0, recall=0, f1_score=0, is_valid=False, message=f'读取验证集失败: {e}')

    p, r, f = _f1(sub, val)
    return dict(precision=p, recall=r, f1_score=f, is_valid=f > 0.0055, message='评估成功')


def evaluate_list(pairs: List[Tuple[int, int]]) -> dict:
    """
    评估任意 N 条 [(user_id, book_id), ...]
    只拿这些用户做真值子集
    """
    if not pairs:
        return dict(precision=0, recall=0, f1_score=0, is_valid=False, message='提交为空')

    sub = pd.DataFrame(pairs, columns=['user_id', 'book_id'])
    users = sub['user_id'].unique().tolist()

    try:
        val = pd.read_csv(VALID_PATH, encoding='ANSI')[['user_id', 'book_id']]\
                .drop_duplicates()\
                .query('user_id in @users')
    except Exception as e:
        return dict(precision=0, recall=0, f1_score=0, is_valid=False, message=f'验证集读取失败: {e}')

    p, r, f = _f1(sub, val)
    return dict(precision=p, recall=r, f1_score=f, is_valid=f > 0.0055, message=f'{len(pairs)} 条评估完成')


# ---------- 快捷封装 ----------
def evaluate_single(u: int, b: int) -> dict:
    return evaluate_list([(u, b)])


def evaluate_triple(u1: int, b1: int, u2: int, b2: int, u3: int, b3: int) -> dict:
    return evaluate_list([(u1, b1), (u2, b2), (u3, b3)])


# ---------- 演示 ----------
if __name__ == '__main__':
    a=pd.read_csv("submission.csv")
    print(evaluate(a))
    print(f"{evaluate(a)['f1_score']:.8f}")