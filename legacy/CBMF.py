import csv
import signal
import subprocess
import sys
from typing import Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
from datetime import datetime
import os
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["PYTHONUNBUFFERED"] = "1"
def setup_logging():
    """设置日志系统"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"content_mf_training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__), log_file


logger, log_file = setup_logging()
logger.info(f"基于内容的矩阵分解 - 日志文件: {log_file}")


class ContentMFConfig:
    """基于内容的矩阵分解配置"""

    # ==================== 模型结构参数 ====================
    MODEL = {
        'factors': 1024,  # 隐向量维度
        'book_vector_dim': 512,  # 书籍向量维度
    }

    # ==================== 训练参数 ====================
    TRAINING = {
        'epochs': 150,
        'batch_size': 4096,
        'learning_rate': 1e-2,
        'weight_decay': 4e-5,
        'random_state': 42,
    }

    # ==================== 数据参数 ====================
    DATA = {
        'file_path': "DataLoad/user_inter_matrix.parquet",  # 交互数据文件路径
        'rating_range': (1, 10),
        'min_user_ratings': 1,
        'min_item_ratings': 1,
    }

    # ==================== 内容特征参数 ====================
    CONTENT = {
        'book_vector_path': "DataLoad/book_vec.parquet",  # 书籍向量文件路径
    }

    # ==================== 其他参数 ====================
    OPTIMIZER = {
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'amsgrad': False,
    }

    EARLY_STOPPING = {
        'enabled': True,
        'patience': 3,
        'delta': 0.001,
        'monitor': 'loss',
    }

    DEVICE = {
        'use_gpu': True,
        'device_id': 0,
    }

    LOGGING = {
        'verbose': True,
        'log_interval': 10,
        'save_model': True,
        'model_path': 'cb_mf_model.pth',
        'plot_loss': True,
    }

    SCHEDULER = {
        'enabled': True,
        'factor': 0.5,
        'patience': 10,
        'min_lr': 1e-6,
    }

    @classmethod
    def get_device(cls):
        if cls.DEVICE['use_gpu'] and torch.cuda.is_available():
            return torch.device(f"cuda:{cls.DEVICE['device_id']}")
        else:
            return torch.device("cpu")

    @classmethod
    def print_config(cls):
        logger.info("=" * 50)
        logger.info("基于内容的矩阵分解配置")
        logger.info("=" * 50)

        for section_name, section in cls.__dict__.items():
            if not section_name.startswith('_') and isinstance(section, dict):
                logger.info(f"\n【{section_name}】")
                for key, value in section.items():
                    logger.info(f"  {key}: {value}")

        logger.info(f"\n训练设备: {cls.get_device()}")
        logger.info("=" * 50)


class ContentBasedMatrixFactorization(nn.Module):
    """
    已知书籍内容向量 → 学出书籍隐向量（可训练）
    再学用户隐向量，最后点积打分
    """
    def __init__(self, n_users, n_items, book_vectors, config=ContentMFConfig):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.n_users = n_users
        self.n_items = n_items
        self.book_vector_dim = book_vectors.shape[1]   # 原始内容维度
        self.latent_dim = config.MODEL['factors'] # 隐空间维度

        # 1. 用户隐向量
        self.user_factors = nn.Embedding(n_users, self.latent_dim)

        # 2. 内容→隐空间映射
        self.content_mapping = nn.Sequential(
            nn.Linear(self.book_vector_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, self.latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.latent_dim)
        )
        self.to(device)
        # 3. 书籍隐向量（可训练 Parameter）
        #    先用映射层把原始向量压到隐空间，再封装成 Parameter
        book_vectors = book_vectors.to(device)
        with torch.no_grad():
            init_item_latent = self.content_mapping(book_vectors)  # [n_items, latent_dim]
        self.item_factors = nn.Parameter(init_item_latent)  # 可训练！

        # 4. 偏置
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Parameter(torch.zeros(n_items, 1))
        self.global_bias = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

        self.interaction_net = nn.Sequential(
            nn.Linear(2 * self.latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )
    def _init_weights(self):
        # 用户向量小随机
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        # 书籍隐向量已用内容初始化，不再额外随机
        nn.init.zeros_(self.item_bias)

    def forward(self, user, item):
        u_vec = self.user_factors(user)  # [B, latent_dim]
        u_b = self.user_bias(user).squeeze()  # [B]
        i_vec = self.item_factors[item]  # [B, latent_dim]
        i_b = self.item_bias[item].squeeze()  # [B]

        score=(u_vec*i_vec).sum(dim=1)

        prediction = self.global_bias + u_b + i_b + score
        return prediction


class ContentBasedMF:
    """基于内容的矩阵分解训练器"""

    def __init__(self, n_users, n_items, book_vectors, config=ContentMFConfig):
        def handler(sig, frame):
            print("\n保存模型中...")
            torch.save(self.model.state_dict(), 'cb_mf_model.pth')
            print("已保存到 interrupt_model.pth")
            exit(0)
        signal.signal(signal.SIGINT, handler)

        self.config = config
        self.device = config.get_device()

        # 将书籍向量转换为tensor并移动到设备
        self.book_vectors = torch.FloatTensor(book_vectors).to(self.device)

        # 创建模型
        self.model = ContentBasedMatrixFactorization(
            n_users, n_items, self.book_vectors, config
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.TRAINING['learning_rate'],
            weight_decay=config.TRAINING['weight_decay']
        )
        self.criterion = nn.MSELoss()
        self.loss_history = []
        self.val_loss_history = []

    def train_model(self, train_data, val_data=None):
        """训练模型"""
        self.model.train()

        # 准备数据
        users = torch.tensor(train_data[:, 0], dtype=torch.long, device=self.device)
        items = torch.tensor(train_data[:, 1], dtype=torch.long, device=self.device)
        ratings = torch.FloatTensor(train_data[:, 2]).to(self.device)

        dataset = TensorDataset(users, items, ratings)
        dataloader = DataLoader(dataset, batch_size=self.config.TRAINING['batch_size'], shuffle=True)

        # 学习率调度器
        scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=80,  # 每30轮下降一次
            gamma=0.5  # 学习率乘以0.5
        )

        best_val_loss = float('inf')
        patience_counter = 0

        epoch_pbar = tqdm(range(self.config.TRAINING['epochs']), desc="训练进度")

        for epoch in epoch_pbar:
            self.model.train()
            epoch_loss = 0.0

            for batch_users, batch_items, batch_ratings in dataloader:
                self.optimizer.zero_grad()

                predictions = self.model(batch_users, batch_items)
                loss = self.criterion(predictions, batch_ratings)

                loss.backward()

                self.optimizer.step()


                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.loss_history.append(avg_loss)
            scheduler.step()
            # 验证集评估
            val_loss = None
            if val_data is not None:
                val_loss = self.evaluate(val_data)
                self.val_loss_history.append(val_loss)

                # 学习率调度


                # 早停检查
                if self.config.EARLY_STOPPING['enabled']:
                    if val_loss < best_val_loss - self.config.EARLY_STOPPING['delta']:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            postfix = {
                'loss': f'{avg_loss:.4f}',
                'val_loss': f'{val_loss:.4f}' if val_loss is not None else 'N/A',
                'lr': f'{current_lr:.2e}'
            }
            if val_loss is not None:
                postfix['val_loss'] = f'{val_loss:.4f}'
                if self.config.EARLY_STOPPING['enabled']:
                    postfix['patience'] = f'{patience_counter}/{self.config.EARLY_STOPPING["patience"]}'

            epoch_pbar.set_postfix(postfix)


        epoch_pbar.close()
        logger.info("训练完成!")
        self.save_model()

    def evaluate(self, test_data):
        """评估模型"""
        self.model.eval()
        with torch.no_grad():
            users = torch.LongTensor(test_data[:, 0]).to(self.device)
            items = torch.LongTensor(test_data[:, 1]).to(self.device)
            ratings = torch.FloatTensor(test_data[:, 2]).to(self.device)

            predictions = self.model(users, items)
            loss = torch.sqrt(self.criterion(predictions, ratings))

            return loss.item()

    def save_model(self, filepath=None):
        """保存模型"""
        if filepath is None:
            filepath = self.config.LOGGING['model_path']

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history,
            'config': self.config,
            'n_users': self.model.n_users,
            'n_items': self.model.n_items,
            'book_vector_dim': self.model.book_vector_dim
        }, filepath)

        logger.info(f"模型已保存到: {filepath}")

with open('id_mappings.json', 'r', encoding='utf-8') as f:
    df = json.load(f)

def find_keys(target_list):
    inner = df['index_to_user']
    # 字符串→int 并摊平
    return [int(k) for t in target_list for k, v in inner.items() if v == t]


def get_all_real_user_ids(mapping_file: str = 'id_mappings.json') -> List:
    """提取所有真实用户ID（来自id_mappings.json）"""
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)

        # 从index_to_user的value中提取（更直接，value就是真实用户ID）
        if 'index_to_user' in mappings and isinstance(mappings['index_to_user'], dict):
            return list(mappings['index_to_user'].values())
        # 或者从user_to_index的key中提取（效果相同）
        elif 'user_to_index' in mappings and isinstance(mappings['user_to_index'], dict):
            return list(mappings['user_to_index'].keys())
        else:
            raise KeyError("映射文件中缺少 'index_to_user' 或 'user_to_index' 字段")

    except Exception as e:
        raise RuntimeError(f"提取真实用户ID失败: {str(e)}")

def load_book_vectors(config=ContentMFConfig):
    """加载书籍向量数据"""
    try:
        df = pd.read_parquet(config.CONTENT['book_vector_path'])
        logger.info(f"书籍向量数据加载成功: {df.shape}")

        # 提取书籍向量
        book_vectors = []
        book_ids = []

        for _, row in df.iterrows():
            book_ids.append(row['book_id'])
            # 将字符串形式的向量转换为numpy数组
            if isinstance(row['book_vector'], str):
                # 处理字符串格式的向量，如 "[0.1, 0.2, 0.3]"
                vector_str = row['book_vector'].strip('[]')
                vector = np.fromstring(vector_str, sep=',')
            elif isinstance(row['book_vector'], list):
                # 如果已经是列表格式
                vector = np.array(row['book_vector'])
            else:
                # 其他格式，尝试直接转换
                vector = np.array(row['book_vector'])

            book_vectors.append(vector)

        book_vectors = np.array(book_vectors)
        logger.info(f"书籍向量形状: {book_vectors.shape}")

        # 创建书籍ID到索引的映射
        book_id_to_index = {book_id: idx for idx, book_id in enumerate(book_ids)}

        return book_vectors, book_id_to_index

    except Exception as e:
        logger.error(f"加载书籍向量失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

def load_interaction_data(config):
    """
    加载用户-物品交互数据
    返回: train_data, val_data, df_reset
        train_data / val_data: shape=(N, 3) 的 numpy 数组，列=[user_idx, item_idx, rating]
    """
    logging.info("开始读取交互文件...")
    df = pd.read_parquet(config.DATA['file_path'])
    logging.info(f"交互数据加载成功: {df.shape}")

    df_reset = df.reset_index(drop=True)
    df_reset.columns = range(df_reset.shape[1])

    # ===== 向量化构造长表 =====
    logging.info("转换交互矩阵为长表（向量化）...")
    stacked = df_reset.stack(dropna=True)
    records = pd.DataFrame({
        'user': stacked.index.get_level_values(0).astype(int),
        'item': stacked.index.get_level_values(1).astype(int),
        'rating': stacked.values.astype(float),
        'time': stacked.index.get_level_values(1).astype(int)
    }).reset_index(drop=True)
    logging.info(f"有效评分数量: {len(records)}")

    # 按用户、按时间排序
    records = records.sort_values(['user', 'time'])

    # leave-last-one-out 划分
    def split_one_user(g, n_keep=0):
        """
        g: 单个用户的 DataFrame（已按时间排好序）
        n_keep: 要留出来做验证的交互条数，默认 1（leave-one-out）
        """
        if len(g) < n_keep + 1:  # 必须至少有一条留给训练
            return g, g.iloc[:0]  # 训练集拿全部，验证集空
        valid = g.tail(n_keep)  # 最晚 n_keep 条做验证
        train = g.drop(valid.index)  # 其余训练
        return train, valid

    logging.info("正在 leave-last-one-out 划分...")
    chunks = records.groupby('user', group_keys=False).apply(split_one_user)
    train_df = pd.concat([c[0] for c in chunks]).reset_index(drop=True)
    val_df   = pd.concat([c[1] for c in chunks]).reset_index(drop=True)

    train_data = train_df[['user', 'item', 'rating']].values
    val_data   = val_df[['user', 'item', 'rating']].values

    logging.info(f"划分完成 — train: {len(train_data)}, val: {len(val_data)}")
    return train_data, val_data, df_reset



def load_and_prepare_content_data(config=ContentMFConfig):
    """加载交互数据和书籍向量"""
    logger.info("开始加载数据...")

    # 加载交互数据
    training_data, val_data,df = load_interaction_data(config)
    if training_data is None:
        logger.error("交互数据加载失败")
        return None, None, None, None, None

    # 加载书籍向量
    book_vectors, book_id_to_index = load_book_vectors(config)
    if book_vectors is None:
        logger.error("书籍向量加载失败")
        return None, None, None, None, None


    logger.info(f"训练集: {len(training_data)}, 验证集: {len(val_data)}")
    return training_data, val_data,  df, book_vectors



def content_main():
    """基于内容的矩阵分解主函数"""
    ContentMFConfig.print_config()

    # 加载数据
    result = load_and_prepare_content_data()
    if result is None:
        logger.error("数据加载失败")
        return None, None, None

    train_data, val_data, df, book_vectors = result

    # 创建模型
    n_users, n_items = df.shape
    n_items = min(n_items, len(book_vectors))  # 确保维度匹配
    logger.info(f"创建模型: {n_users}用户, {n_items}物品, 书籍向量维度: {book_vectors.shape[1]}")

    # 更新配置中的书籍向量维度
    ContentMFConfig.MODEL['book_vector_dim'] = book_vectors.shape[1]

    content_mf = ContentBasedMF(n_users, n_items, book_vectors)

    # 训练模型
    logger.info("开始训练基于内容的矩阵分解模型...")

    content_mf.train_model(train_data, val_data)


    return content_mf, df, book_vectors


def load_book_vectors_simple(file_path):
    """简化版的书籍向量加载函数"""
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"📚 加载书籍数据: {len(df)} 本书籍")

        book_vectors = []
        book_ids = []

        for _, row in df.iterrows():
            book_ids.append(row['book_id'])

            # 处理向量数据
            vector_data = row['book_vector']
            if isinstance(vector_data, str):
                # 字符串格式: "[0.1, 0.2, 0.3]"
                vector_str = vector_data.strip('[]')
                vector = np.fromstring(vector_str, sep=',', dtype=np.float32)
            elif isinstance(vector_data, list):
                vector = np.array(vector_data, dtype=np.float32)
            else:
                vector = np.array(vector_data, dtype=np.float32)

            book_vectors.append(vector)

        book_vectors = np.array(book_vectors)
        book_id_to_index = {book_id: idx for idx, book_id in enumerate(book_ids)}

        logger.info(f"✅ 书籍向量加载成功: {book_vectors.shape}")
        return book_vectors, book_id_to_index

    except Exception as e:
        logger.error(f"❌ 书籍向量加载失败: {e}")
        return None, None


def batch_recommend(user_ids, model_path="cb_mf_model.pth", book_vector_path="DataLoad/book_vec.parquet",
                    n_recommend=1):
    """
    批量为用户生成推荐（优化版）
    """
    logger.info(f"👥 开始批量推荐: {len(user_ids)} 个用户")

    # 1. 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"💻 使用设备: {device}")

    # 2. 加载模型和书籍向量
    logger.info(f"🔍 加载模型检查点: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    logger.info(f"📖 加载书籍向量: {book_vector_path}")
    book_vectors, book_id_to_index = load_book_vectors_simple(book_vector_path)
    if book_vectors is None:
        raise ValueError("无法加载书籍向量")

    # 3. 转换书籍向量为张量并移动到设备
    logger.info("🔄 转换书籍向量为PyTorch张量...")
    if isinstance(book_vectors, np.ndarray):
        book_vectors_tensor = torch.FloatTensor(book_vectors).to(device)
    else:
        book_vectors_tensor = book_vectors.to(device)
    os.system('python tem2.py')
    # 4. 初始化模型
    config = checkpoint.get('config', ContentMFConfig)
    n_users = checkpoint['n_users']
    n_items = checkpoint['n_items']
    logger.info(f"📊 模型参数: {n_users}用户, {n_items}物品")

    model = ContentBasedMatrixFactorization(n_users, n_items, book_vectors_tensor, config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 5. 预处理映射和物品索引
    index_to_book_id = {idx: book_id for book_id, idx in book_id_to_index.items()}
    all_item_indices = torch.arange(n_items, dtype=torch.long, device=device)  # 预生成所有物品索引（关键优化）
    user_ids_idx = find_keys(user_ids)  # 用户索引列表

    results = {}

    logger.info("🚀 开始批量计算推荐分数...")
    with torch.no_grad():
        # 批量处理用户（若用户数量大，可再分用户批次）
        for user_id, user_idx in tqdm(zip(user_ids, user_ids_idx),
                                      total=len(user_ids),
                                      desc="批量处理用户"):
            # 生成用户张量（形状: [1]）
            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)

            # 核心优化：一次性计算该用户对所有物品的评分（替代循环）
            # 扩展用户索引以匹配物品数量（形状: [n_items]）
            user_batch = user_tensor.repeat(n_items)  # 每个物品对应同一个用户
            # 批量计算评分（形状: [n_items]）
            scores = model(user_batch, all_item_indices)  # 直接传入所有物品索引

            # 转换为numpy并排序（取Top-N）
            scores_np = scores.cpu().numpy()  # 仅在排序时转移到CPU
            top_indices = scores_np.argsort()[::-1][:n_recommend]  # 降序取前n_recommend

            # 构建推荐结果
            user_recommendations = [
                (index_to_book_id.get(idx, f"book_{int(idx + 1)}"), round(scores_np[idx], 4))
                for idx in top_indices
            ]
            results[user_id] = user_recommendations

    logger.info(f"✅ 批量推荐完成: {len(results)} 个用户")
    return results

# 使用示例
if __name__ == "__main__":
    # 训练基于内容的模型
    # IsTraining = 1
    # if IsTraining:
    # content_main()
    # else:
    input=get_all_real_user_ids()
    result=batch_recommend(input)

    print("数据已成功保存到 submission.csv")
