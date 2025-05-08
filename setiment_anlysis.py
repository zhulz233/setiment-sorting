import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, f1_score
import re
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# 1. 数据预处理
# ----------------------------

# 加载预训练词向量
def load_word_vectors():
    word2vec = KeyedVectors.load_word2vec_format('wiki_word2vec_50.bin', binary=True)
    return word2vec

# 构建词汇表和嵌入矩阵
def build_vocab(word2vec, max_words=None):
    vocab = {'<pad>': 0, '<unk>': 1}
    embeddings = np.zeros((2, 50))  # 初始pad和unk
    
    # 添加预训练词向量
    for word in word2vec.key_to_index:
        vocab[word] = len(vocab)
        embeddings = np.vstack([embeddings, word2vec[word]])
    
    # 添加未登录词处理
    vocab['<unk>'] = 1  # 已存在，但需要明确保留
    
    return vocab, embeddings

# 统计最大句子长度
def get_max_len(file_path):
    max_len = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            _, sentence = line.strip().split(' ', 1)
            words = sentence.split()
            max_len = max(max_len, len(words))
    return max_len

# 自定义数据集
class SentimentDataset(Dataset):
    def __init__(self, file_path, word2idx, max_len):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                label, sentence = line.strip().split(' ', 1)
                words = sentence.split()
                # 转换为索引序列
                indices = [word2idx.get(word, word2idx['<unk>']) for word in words[:max_len]]
                # 填充
                indices += [word2idx['<pad>']] * (max_len - len(indices))
                self.data.append((indices, int(label)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

# ----------------------------
# 2. 模型定义
# ----------------------------

# CNN模型
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) 
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes)*num_filters, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, emb_dim]
        x = x.unsqueeze(1)     # [batch, 1, seq_len, emb_dim]
        x = [conv(x).squeeze(3) for conv in self.convs]  # [batch, num_filters, seq_len-k+1]
        x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# LSTM模型
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                          bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x)
        last_hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1) if self.lstm.bidirectional else hn[-1,:,:]
        return self.fc(self.dropout(last_hidden))

# MLP模型
class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim*200, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ----------------------------
# 3. 训练与评估
# ----------------------------

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, patience=3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        val_acc, val_f1 = evaluate(model, val_loader)
        print(f'Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, '
              f'Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}')
        
        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                model.load_state_dict(torch.load('best_model.pth'))
                return model
    
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1

# ----------------------------
# 4. 主程序
# ----------------------------

def main():
    # 参数配置
    params = {
        'vocab_size': None,
        'embedding_dim': 50,
        'max_len': 200,
        'batch_size': 64,
        'epochs': 15,
        'lr': 0.001
    }
    
    # 1. 数据准备
    word2vec = load_word_vectors()
    vocab, embeddings = build_vocab(word2vec)
    params['vocab_size'] = len(vocab)
    
    # 2. 创建数据集
    max_len = params['max_len']
    train_data = SentimentDataset('train.txt', vocab, max_len)
    val_data = SentimentDataset('validation.txt', vocab, max_len)
    test_data = SentimentDataset('test.txt', vocab, max_len)
    
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params['batch_size'])
    test_loader = DataLoader(test_data, batch_size=params['batch_size'])
    
    # 3. 模型训练与评估
    models = {
        'CNN': CNN(params['vocab_size'], params['embedding_dim'], num_filters=100, 
                   filter_sizes=[3,4,5], num_classes=2),
        'LSTM': LSTM(params['vocab_size'], params['embedding_dim'], hidden_dim=128,
                    num_layers=2, num_classes=2),
        'MLP': MLP(params['vocab_size'], params['embedding_dim'], hidden_dim=256,
                  num_classes=2)
    }
    
    results = {}
    for name, model in models.items():
        print(f'\nTraining {name} model...')
        model.to(device)
        model.embedding.weight.data.copy_(torch.tensor(embeddings))
        model.embedding.weight.requires_grad = True  # 允许微调
        
        # 训练模型
        model = train_model(model, train_loader, val_loader, 
                           epochs=params['epochs'], lr=params['lr'])
        
        # 评估模型
        test_acc, test_f1 = evaluate(model, test_loader)
        results[name] = (test_acc, test_f1)
        print(f'{name} Test Results - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}')
    
    # 4. 结果对比
    print('\nModel Comparison:')
    for model_name, (acc, f1) in results.items():
        print(f'{model_name}: Accuracy={acc:.4f}, F1={f1:.4f}')

if __name__ == '__main__':
    main()
