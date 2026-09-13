import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
from datetime import datetime
import os
import json
import torch.nn.functional as F
import warnings
warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns",
    category=FutureWarning
)

class MyConfig:
    """
    矩阵分解推荐算法的超参数配置类
    包含模型参数、训练参数、数据参数等所有可配置选项
    """

    # ==================== 模型结构参数 ====================
    MODEL = {
        'n_factors': 20,  # 隐向量维度，控制模型复杂度
        # 说明：隐向量维度越大，模型表达能力越强，但可能过拟合
        # 推荐范围：10-100，根据数据规模调整
    }

    # ==================== 训练参数 ====================
    TRAINING = {
        'epochs': 250,  # 训练轮数
        'batch_size': 2048,  # 批处理大小
        'learning_rate': 2e-4,  # 学习率
        'weight_decay': 0.001,  # L2正则化系数（权重衰减）
        'random_state': 42,  # 随机种子（保证结果可复现）

        # 说明：
        # - epochs: 根据数据大小调整，大数据集可减少轮数
        # - batch_size: 根据GPU内存调整，越大训练越快但内存消耗大
        # - learning_rate: 关键参数，太大导致震荡，太小收敛慢
        # - weight_decay: 防止过拟合，值越大正则化越强
    }

    # ==================== 优化器参数 ====================
    OPTIMIZER = {
        'betas': (0.9, 0.999),  # Adam优化器的动量参数
        'eps': 1e-8,  # 数值稳定项
        'amsgrad': True,  # 是否使用AMSGrad变体

        # 说明：
        # - betas: (一阶矩衰减率, 二阶矩衰减率)，通常不需要修改
        # - eps: 防止除零的小常数
        # - amsgrad: 通常为False，True可能在某些情况下有更好效果
    }

    # ==================== 数据参数 ====================
    DATA = {
        'file_path': "DataLoad/user_inter_matrix.parquet",  # 数据文件路径
        'rating_range': (1, 10),  # 评分范围，用于数据验证
        # 说明：
        # - rating_range: 根据实际评分系统调整
        # - min_xxx_ratings: 过滤数据，提高模型质量
    }

    # ==================== 早停参数 ====================
    EARLY_STOPPING = {
        'enabled': False,  # 是否启用早停
        'patience': 1,  # 容忍轮数（验证集损失不改善的轮数）
        'delta': 0.001,  # 最小改善阈值
        'monitor': 'loss',  # 监控指标：'loss' 或 'rmse'

        # 说明：
        # - enabled: 小数据集建议开启，大数据集可关闭
        # - patience: 通常5-10轮，太大可能过拟合，太小可能提前停止
        # - delta: 小于此值的改善视为无改善
    }

    # ==================== 评估参数 ====================
    EVALUATION = {
        'top_k': 1,  # 推荐列表长度
        'metrics': ['rmse', 'mae'],  # 评估指标
        'verbose': True,  # 是否显示详细评估信息

        # 说明：
        # - top_k: 生成推荐的数量，通常5-20
        # - metrics: 评估指标列表，可扩展其他指标
    }

    # ==================== 设备参数 ====================
    DEVICE = {
        'use_gpu': True,  # 是否使用GPU
        'device_id': 0,  # GPU设备ID（多GPU时使用）

        # 说明：
        # - use_gpu: 有GPU时设为True加速训练
        # - device_id: 多GPU时指定使用哪个GPU
    }

    # ==================== 日志参数 ====================
    LOGGING = {
        'verbose': True,  # 是否显示训练进度
        'log_interval': 100,  # 日志打印间隔（轮数）
        'save_model': True,  # 是否保存训练好的模型
        'model_path': 'fin_model.pth',  # 模型保存路径
        'plot_loss': True,  # 是否绘制损失曲线

        # 说明：
        # - log_interval: 每多少轮打印一次日志
        # - save_model: 训练完成后保存模型文件
    }
    # ==================== 学习率调度参数 ====================
    SCHEDULER = {
        'enabled': True,  # 是否启用学习率调度
        'factor': 0.5,  # 损失不改善时学习率减半
        'patience': 10,
        'min_lr': 1e-6,  # 最小学习率
    }

    @classmethod
    def get_device(cls):
        """获取训练设备（CPU或GPU）"""
        if cls.DEVICE['use_gpu'] and torch.cuda.is_available():
            return torch.device(f"cuda:{cls.DEVICE['device_id']}")
        else:
            return torch.device("cpu")

    @classmethod
    def print_config(cls):
        """打印当前配置"""
        logger.info("=" * 50)
        logger.info("矩阵分解推荐算法配置")
        logger.info("=" * 50)

        for section_name, section in cls.__dict__.items():
            if not section_name.startswith('_') and isinstance(section, dict):
                logger.info(f"\n【{section_name}】")
                for key, value in section.items():
                    logger.info(f"  {key}: {value}")

        device = cls.get_device()
        logger.info(f"\n训练设备: {device}")
        logger.info("=" * 50)

def setup_logging():
    """设置日志系统"""
    # 创建logs目录（如果不存在）
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mf_training_{timestamp}.log")

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 文件处理器
            logging.StreamHandler()  # 控制台处理器
        ]
    )

    return logging.getLogger(__name__), log_file


# 在文件开始处设置日志
logger, log_file = setup_logging()
logger.info(f"日志文件: {log_file}")


class MatrixFactorization(nn.Module):

    def __init__(self, n_users, n_items, config=MyConfig):
        super().__init__()
        self.config = config
        self.n_factors = config.MODEL['n_factors']

        # 嵌入层
        self.user_factors = nn.Embedding(n_users, self.n_factors)
        self.item_factors = nn.Embedding(n_items, self.n_factors)

        # 偏置项
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))

        # 计算总特征数：基础向量(2*k) + 2个核心特征
        total_features = 2 * self.n_factors + 2

        # 最简单的网络
        self.score_head = nn.Sequential(
            nn.Linear(total_features, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        self._init_weights()
    def _init_weights(self):
        """初始化权重"""
        # 嵌入层初始化
        self.user_factors.weight.data.uniform_(0, 0.1)
        self.item_factors.weight.data.uniform_(0, 0.1)
        self.user_bias.weight.data.uniform_(0, 0.01)
        self.item_bias.weight.data.uniform_(0, 0.01)

    def forward(self, user, item):
        # 1. 获取基础嵌入向量
        user_vec = self.user_factors(user)  # [batch_size, n_factors]
        item_vec = self.item_factors(item)  # [batch_size, n_factors]

        # 2. 只保留最核心的两个特征
        # 特征1: 点积相似度（最重要的特征）
        dot_product = torch.sum(user_vec * item_vec, dim=1, keepdim=True)  # [B, 1]

        # 特征2: 余弦相似度（角度信息）
        cosine_sim = F.cosine_similarity(user_vec, item_vec, dim=1).unsqueeze(1)  # [B, 1]

        # 3. 直接拼接基础向量和核心特征
        feat = torch.cat([user_vec, item_vec, dot_product, cosine_sim], dim=1)  # [B, 2*k + 2]

        # 4. 通过简单网络
        score = self.score_head(feat).squeeze(1)  # [B]

        # 5. 计算偏置项
        user_b = self.user_bias(user).squeeze()  # [batch_size]
        item_b = self.item_bias(item).squeeze()  # [batch_size]

        # 6. 最终预测
        pred = self.global_bias + user_b + item_b + score
        return pred


class PyTorchMF:
    """PyTorch矩阵分解训练器"""

    def __init__(self, n_users, n_items, config=MyConfig):
        self.config = config
        self.device = config.get_device()
        self.final_model_path = "final_model.pth"  # 训练结束时的最终模型
        # 创建模型并移动到设备
        self.model = MatrixFactorization(n_users, n_items, config).to(self.device)

        # 配置优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.TRAINING['learning_rate'],
            weight_decay=config.TRAINING['weight_decay'],
            betas=config.OPTIMIZER['betas'],
            eps=config.OPTIMIZER['eps'],
            amsgrad=config.OPTIMIZER['amsgrad']
        )

        self.criterion = nn.MSELoss()
        self.loss_history = []
        self.val_rmse_history = []

    def train_model(self, train_matrix):
        """
        输入：宽表矩阵 shape = (n_users, n_items)
        内部：0 值舍弃，转成长表 (user_id, item_id, rating)
        """

        device = next(self.model.parameters()).device

        # 1. 构造训练集 DataLoader
        users = torch.LongTensor(train_matrix[:, 0]).to(device)
        items = torch.LongTensor(train_matrix[:, 1]).to(device)
        ratings = torch.FloatTensor(train_matrix[:, 2]).to(device)
        train_loader = DataLoader(TensorDataset(users, items, ratings),
                                  batch_size=self.config.TRAINING['batch_size'],
                                  shuffle=True)

        epochs = self.config.TRAINING['epochs']

        # 2. 创建单个进度条（不显示批次进度）
        pbar = tqdm(range(epochs), desc="训练进度", unit="epoch")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3,
            min_lr=1e-6, verbose=False)

        # 3. 训练主循环
        for epoch in pbar:
            # 训练阶段（不显示批次进度条）
            self.model.train()
            epoch_loss = 0.0

            for bu, bi, br in train_loader:
                self.optimizer.zero_grad()
                pred = self.model(bu, bi)
                loss = self.criterion(pred, br)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            self.loss_history.append(avg_train_loss)

            # 在训练循环中添加：
            scheduler.step(avg_train_loss)
            # 更新进度条信息（不创建新进度条）
            lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'epoch': f'{epoch + 1}/{epochs}',
                'loss': f'{avg_train_loss:.4f}',
                'lr': f'{lr:.2e}'
            })

            # 日志记录
            if (epoch + 1) % self.config.LOGGING.get('log_interval', 1) == 0:
                logger.info(f"Epoch {epoch + 1:03d}/{epochs} | "
                            f"TrainLoss={avg_train_loss:.4f} | "
                            f"LR={lr:.2e}")

        # 4. 保存最终模型
        self.save_model("fin_model.pth")

        # 5. 关闭进度条
        pbar.close()

        logger.info(f"✅ 训练完成 | 最终训练损失={avg_train_loss:.4f}")
        return {"final_train_loss": avg_train_loss}

    def save_model(self, filepath=None):
        """保存模型 - 修复版本"""
        if filepath is None:
            filepath = self.config.LOGGING['model_path']

        n_users = self.model.user_factors.num_embeddings
        n_items = self.model.item_factors.num_embeddings
        n_factors = self.model.n_factors

        torch.save({
            'model_state_dict': self.model.state_dict(),  # 保存整个模型的状态
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'val_rmse_history': getattr(self, 'val_rmse_history', []),
            'config': self.config,
            # 添加这些关键参数！
            'n_users': n_users,
            'n_items': n_items,
            'n_factors': n_factors
        }, filepath)
        logger.info(f"模型已保存到: {filepath}")
        logger.info(f"保存参数: {n_users} 用户, {n_items} 物品, {n_factors} 隐因子")

    def evaluate(self, val_long):
        self.model.eval()
        val_users = torch.LongTensor(val_long[:, 0]).to(self.device)
        val_items = torch.LongTensor(val_long[:, 1]).to(self.device)
        val_truth = torch.FloatTensor(val_long[:, 2]).to(self.device)

        with torch.no_grad():
            pred = self.model(val_users, val_items)
            rmse = torch.sqrt(F.mse_loss(pred, val_truth))
        return rmse.item()
    def plot_loss(self):
        """绘制损失曲线并保存图像（增强版）"""
        if not self.config.LOGGING['plot_loss']:
            return

        import os
        from datetime import datetime

        # 创建图像保存目录
        plot_dir = "plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            logger.info(f"创建图像目录: {plot_dir}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # 创建更大的图像以便更清晰
            plt.figure(figsize=(15, 6))

            # 子图1：训练损失
            plt.subplot(1, 2, 1)
            plt.plot(self.loss_history, 'b-', linewidth=2, label='Training Loss')
            plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('MSE Loss', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()

            # 添加最后损失值标注
            if self.loss_history:
                final_loss = self.loss_history[-1]
                plt.annotate(f'Final: {final_loss:.4f}',
                             xy=(len(self.loss_history) - 1, final_loss),
                             xytext=(10, 10), textcoords='offset points',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

            # 子图2：验证损失（如果有）
            if self.val_rmse_history:
                plt.subplot(1, 2, 2)
                plt.plot(self.val_rmse_history, 'r-', linewidth=2, label='Validation RMSE')
                plt.title('Validation RMSE Curve', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('RMSE', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend()

                # 添加最后RMSE值标注
                final_rmse = self.val_rmse_history[-1]
                plt.annotate(f'Final: {final_rmse:.4f}',
                             xy=(len(self.val_rmse_history) - 1, final_rmse),
                             xytext=(10, 10), textcoords='offset points',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
            else:
                # 如果没有验证损失，显示训练损失的详细视图
                plt.subplot(1, 2, 2)
                plt.plot(self.loss_history, 'g-', linewidth=2, alpha=0.7, label='Training Loss (Detailed)')
                plt.title('Training Loss (Last 50 Epochs)', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('MSE Loss', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend()

                # 如果轮数较多，只显示最后50轮
                if len(self.loss_history) > 50:
                    plt.xlim(len(self.loss_history) - 50, len(self.loss_history) - 1)

            plt.tight_layout()

            # 保存高分辨率图像
            plot_file = os.path.join(plot_dir, f"loss_curve_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logger.info(f"✅ 损失曲线已保存到: {plot_file}")

            # 同时保存PDF版本（矢量图，更清晰）
            pdf_file = os.path.join(plot_dir, f"loss_curve_{timestamp}.pdf")
            plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"✅ 损失曲线PDF版本已保存到: {pdf_file}")

            # 记录图像统计信息
            logger.info(f"📊 损失曲线统计:")
            logger.info(f"   - 训练轮数: {len(self.loss_history)}")
            if self.loss_history:
                logger.info(f"   - 最终训练损失: {self.loss_history[-1]:.6f}")
            if self.val_rmse_history:
                logger.info(f"   - 最终验证RMSE: {self.val_rmse_history[-1]:.6f}")

        except Exception as e:
            logger.error(f"❌ 保存损失曲线失败: {e}")
            import traceback
            logger.error(traceback.format_exc())



def load_and_prepare_data(config=MyConfig):
    logger.info("加载数据...")

    try:
        df = pd.read_parquet(config.DATA['file_path'])
        logger.info(f"数据加载成功！形状: {df.shape}")
    except Exception as e:
        logger.info(f"数据加载失败: {e}")
        return None, None, None, None

    # 快速分析数据
    total_cells = df.shape[0] * df.shape[1]
    non_null_count = df.notna().sum().sum()
    sparsity = 1 - (non_null_count / total_cells)

    logger.info(f"总单元格数: {total_cells:,}")
    logger.info(f"非空评分数: {non_null_count:,}")
    logger.info(f"稀疏度: {sparsity:.4f} ({sparsity*100:.2f}%)")

    logger.info("准备训练数据（快速方法）...")

    # 重置索引，确保索引是连续的整数
    df_reset = df.reset_index(drop=True)
    df_reset.columns = range(len(df_reset.columns))

    # 使用stack获取所有非空值 - 比双重循环快得多
    stacked = df_reset.stack()

    if len(stacked) == 0:
        logger.info("警告：没有找到有效评分数据！")
        return None, None, None, None

    # 转换为训练数据格式 [user_index, item_index, rating]
    training_data = []
    for (user_idx, item_idx), rating in tqdm(stacked.items(), desc="处理评分数据", total=len(stacked)):
        training_data.append([user_idx, item_idx, float(rating)])

    training_data = np.array(training_data)
    logger.info(f"有效评分数量: {len(training_data)}")

    # 分割数据
    if config.TRAINING['test_size'] > 0:
        train_data, test_data = train_test_split(
            training_data,
            test_size=config.TRAINING['test_size'],
            random_state=config.TRAINING['random_state']
        )

        logger.info(f"训练集大小: {len(train_data)}")
        logger.info(f"测试集大小: {len(test_data)}")

        return train_data, None, test_data, df
    else:
        logger.info(f"使用全部数据训练，大小: {len(training_data)}")
        return training_data, None, None, df


def safe_main():
    """安全的主训练函数，处理各种返回值情况"""
    # 打印配置
    MyConfig.print_config()


    # 创建模型
    train_mat, val_mat, df = load_interaction_data(config=MyConfig)  # 返回矩阵
    n_users, n_items = df.shape
    logger.info(f"\n创建模型: {n_users} 用户, {n_items} 物品")

    mf_model = PyTorchMF(n_users, n_items)

    # 训练模型
    logger.info("\n开始训练...")

    mf_model.train_model(train_mat)

    # 绘制损失曲线
    mf_model.plot_loss()

    return mf_model, df


def main():
    """主训练函数 - 兼容旧版本调用"""
    return safe_main()


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
    def split_one_user(g, n_keep=2):
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


def generate_recommendations_for_all_users(model_path='fin_model.pth', top_k=100, output_file='submission.csv'):
    """
    为所有用户生成推荐并保存为CSV文件（使用真实的user_id和book_id）
    """
    print(f"🎯 开始为所有用户生成推荐，每个用户推荐 {top_k} 个物品...")

    try:
        # 1. 加载模型
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ 模型加载成功")

        # 2. 获取模型参数
        state_dict = checkpoint['model_state_dict']
        n_users = state_dict['user_factors.weight'].shape[0]
        n_items = state_dict['item_factors.weight'].shape[0]
        config = checkpoint.get('config', MyConfig)

        print(f"📊 模型信息: {n_users} 用户, {n_items} 物品")

        # 3. 创建模型实例
        model = MatrixFactorization(n_users, n_items, config)
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ 模型初始化完成")

        # 4. 加载ID映射文件

        with open('id_mappings.json', 'r', encoding='utf-8') as f:
            mappings = json.load(f)
            print("✅ ID映射文件加载成功")

            # 获取用户映射：真实user_id -> 索引
            user_to_index = mappings.get('user_to_index', {})
            # 创建反向映射：索引 -> 真实user_id
            index_to_user = {idx: user_id for user_id, idx in user_to_index.items()}

            # 获取物品映射：真实book_id -> 索引
            item_to_index = mappings.get('item_to_index', {})
            # 创建反向映射：索引 -> 真实book_id
            index_to_item = {idx: item_id for item_id, idx in item_to_index.items()}

            print(f"👥 用户映射: {len(index_to_user)} 个用户")
            print(f"📚 物品映射: {len(index_to_item)} 个物品")


        # 5. 生成推荐并保存
        print(f"\n📈 为 {n_users} 个用户生成推荐...")

        total_recommendations = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("user_id,book_id\n")

            for user_index in tqdm(range(n_users), desc="处理用户"):
                try:
                    # 获取真实的user_id
                    real_user_id = index_to_user.get(user_index, str(user_index))

                    user_tensor = torch.LongTensor([user_index])
                    all_scores = []

                    # 批量处理提升性能
                    batch_size = 8000
                    for start_idx in range(0, n_items, batch_size):
                        end_idx = min(start_idx + batch_size, n_items)
                        batch_items = torch.LongTensor(range(start_idx, end_idx))
                        # 修复推荐生成部分
                        batch_users = user_tensor.repeat(batch_items.size(0))  # 形状: [batch_size]
                        with torch.no_grad():
                            batch_scores = model(batch_users, batch_items)

                        for j, score in enumerate(batch_scores.tolist()):
                            item_index = start_idx + j
                            all_scores.append((item_index, score))

                    # 排序取top_k
                    all_scores.sort(key=lambda x: x[1], reverse=True)

                    # 写入文件（使用真实的user_id和book_id）
                    for item_index, score in all_scores[:top_k]:
                        real_book_id = index_to_item.get(item_index, str(item_index))
                        f.write(f"{real_user_id},{int(real_book_id)+1}\n")
                        total_recommendations += 1

                except Exception as e:
                    print(f"⚠️ 用户索引 {user_index} 失败: {e}")
                    continue

        # 6. 验证结果
        print(f"\n✅ 推荐生成完成！共生成 {total_recommendations:,} 条推荐")

        # 读取并显示文件信息
        try:
            df = pd.read_csv(output_file)
            print(f"📁 文件: {output_file}")
            print(f"📊 文件统计:")
            print(f"   - 总行数: {len(df):,}")
            print(f"   - 用户数量: {df['user_id'].nunique()}")
            print(f"   - 物品数量: {df['book_id'].nunique()}")
            print(f"   - 每用户推荐数: {len(df) // df['user_id'].nunique()}")

            # 显示前几行
            print(f"\n📋 文件前5行示例:")
            print(df.head())

            # 显示一些用户ID示例
            unique_users = df['user_id'].unique()[:3]
            print(f"👥 用户ID示例: {list(unique_users)}")

            # 显示一些物品ID示例
            unique_items = df['book_id'].unique()[:3]
            print(f"📚 物品ID示例: {list(unique_items)}")

        except Exception as e:
            print(f"⚠️ 文件读取失败: {e}")

        return True

    except Exception as e:
        print(f"❌ 生成推荐失败: {e}")
        import traceback
        traceback.print_exc()
        return False




# 使用示例
if __name__ == "__main__":

    IsTraining = 0
    if IsTraining:
        main()
    else:
        generate_recommendations_for_all_users(top_k=1)
