import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import json
import sys
from tqdm import tqdm
import qlib
from get_data import get_yaml_config, get_dataset
from model import SimpleTransformerModel
# 添加当前目录到Python路径，以便可以导入base_dataset模块
sys.path.append(os.path.dirname(__file__))

#from base_dataset import StockDataset_min_v2, config, get_train_test_vaild,get_train_test_valid_easy


class SequenceModelTrainer:
    """序列模型训练器类，适配StockDataset_min_v2数据集"""
    
    def __init__(self, model: nn.Module, config: Dict):
        """
        初始化训练器
        
        Args:
            model: PyTorch模型
            config: 训练配置字典
        """
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        # 优化器和损失函数
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.0)
        )
        
        self.criterion = nn.MSELoss() if config.get('task_type', 'regression') == 'regression' else nn.CrossEntropyLoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # 创建保存目录
        self.save_dir = config.get('save_dir', 'saved_models')
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # 使用tqdm显示训练进度
        pbar = tqdm(train_loader, desc="训练", leave=False)
        
        for batch_idx, sequences in enumerate(pbar):
            x = sequences[:,:,:-1].to(self.device)
            x = torch.rand(x.shape, device=self.device)
            print(x)
            # x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            # x = (x - x_mean) / (x_std + 1e-5)
            # x = np.clip(x, -5, 5)
            targets = sequences[:,-1,-1].to(self.device)
            # 前向传播
            # print("x and targets:")
            # print(x)
            self.optimizer.zero_grad()
            outputs = self.model(x)
            
            # 计算损失
            loss = self.criterion(outputs.squeeze(), targets)
            # print("outputs and targets:")
            # print(outputs)
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条描述
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss/num_batches:.6f}'
            })
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # 使用tqdm显示验证进度
        pbar = tqdm(val_loader, desc="验证", leave=False)
        
        with torch.no_grad():
            for sequences in pbar:
                x = sequences[:,:,:-1].to(self.device)
                # x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
                # x = (x - x_mean) / (x_std + 1e-5)
                # x = np.clip(x, -5, 5)
                targets = sequences[:,-1,-1].to(self.device)    
                
                outputs = self.model(x)
                loss = self.criterion(outputs.squeeze(), targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # 更新进度条描述
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'avg_loss': f'{total_loss/num_batches:.6f}'
                })
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None) -> Dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            训练结果字典
        """
        epochs = self.config.get('epochs', 100)
        patience = self.config.get('patience', 10)
        
        print(f"开始训练，设备: {self.device}")
        print(f"训练配置: {self.config}")
        
        no_improvement_count = 0
        
        # 使用tqdm显示epoch进度
        epoch_pbar = tqdm(range(epochs), desc="Epochs")
        
        for epoch in epoch_pbar:
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = 0.0
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
            
            # 更新epoch进度条描述
            if val_loader is not None:
                epoch_pbar.set_postfix({
                    'train_loss': f'{train_loss:.6f}',
                    'val_loss': f'{val_loss:.6f}',
                    'best_val': f'{self.best_val_loss:.6f}'
                })
            else:
                epoch_pbar.set_postfix({
                    'train_loss': f'{train_loss:.6f}',
                    'best_val': f'{self.best_val_loss:.6f}'
                })
            
            # 早停和模型保存
            if val_loader is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                no_improvement_count = 0
                
                # 保存最佳模型
                self.save_model(f'best_model_epoch_{epoch+1}.pth')
                print(f'\n最佳模型已保存，验证损失: {val_loss:.6f}')
            else:
                no_improvement_count += 1
            
            # 早停检查
            if no_improvement_count >= patience:
                print(f'\n早停触发，在epoch {epoch+1}停止训练')
                break
        
        # 关闭进度条
        epoch_pbar.close()
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'final_epoch': epoch + 1
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """评估模型性能"""
        self.model.eval()
        predictions = []
        actuals = []
        total_loss = 0
        
        # 使用tqdm显示评估进度
        pbar = tqdm(test_loader, desc="评估")
        
        with torch.no_grad():
            for sequences, targets, info in pbar:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs.squeeze(), targets)
                total_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
                
                # 更新进度条描述
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'avg_loss': f'{total_loss/(len(predictions)/len(targets)):.6f}'
                })
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # 计算评估指标
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'loss': total_loss / len(test_loader),
            'predictions': predictions,
            'actuals': actuals
        }
    
    def save_model(self, filename: str):
        """保存模型"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, filepath)
        
        # 保存训练配置
        config_path = os.path.join(self.save_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_model(self, filename: str):
        """加载模型"""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']

# 辅助函数：创建数据加载器



# 示例使用代码
if __name__ == "__main__":
    # 示例：创建一个简单的LSTM模型，适配StockDataset_min_v2的输出格式
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(SimpleLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步
            return output
    
    # 创建模型和训练器
    # 根据StockDataset_min_v2的特征维度设置输入大小
    model = SimpleLSTM(input_size=5, hidden_size=64, num_layers=2, output_size=1)  # feature_columns_min_v2有8个特征
    
    config = {
        'learning_rate': 0.001,
        'batch_size': 512,  # 增大批次大小，提高GPU利用率
        'epochs': 50,
        'patience': 10,
        'grad_clip': 1.0,
        'task_type': 'regression',
        'save_dir': 'saved_models'
    }
    
    # 使用更大的Transformer模型，增加GPU计算量
    #model = SimpleTransformerModel(input_size=5, d_model=128, nhead=8, num_layers=4, seq_len=10)
    print(model)
    trainer = SequenceModelTrainer(model, config)
    
    # 使用StockDataset_min_v2创建数据集
    # 这里需要你提供实际的股票数据文件路径
    try:
        # 示例：加载股票数据
        qlib.init(provider_uri ="~/.qlib/qlib_data/my_data_2019/", region="cn")
        config = get_yaml_config()
        dataset = get_dataset(config)
        #stock_data  = stock_data[:5000]
        train_df,test_df,vaild_df = dataset.prepare(["train","test","valid"])
        #train_df,test_df,vaild_df = get_train_test_valid_easy(stock_data)
        # 创建训练和验证数据集
        # train_dataset = StockDataset_min_v2(train_df)
        # val_dataset = StockDataset_min_v2(vaild_df)
        # 创建数据加载器
        train_loader = DataLoader(train_df, batch_size=512, shuffle=True,num_workers=4,pin_memory=True)
        val_loader = DataLoader(vaild_df, batch_size=512, shuffle=False,num_workers=4,pin_memory=True)
        
        # 训练模型
        results = trainer.train(train_loader, val_loader)
        
        # 评估模型
        evaluation = trainer.evaluate(val_loader)
        print(f"评估结果: MSE={evaluation['mse']:.4f}, MAE={evaluation['mae']:.4f}, RMSE={evaluation['rmse']:.4f}")
        
        # 保存最终模型
        trainer.save_model('final_model.pth')
        print("训练完成！")
        
    except FileNotFoundError:
        print("股票数据文件未找到，请检查文件路径")
        print("你可以修改文件路径或使用自己的数据")
    except Exception as e:
        print(f"数据加载错误: {e}")
        print("请确保数据格式正确")