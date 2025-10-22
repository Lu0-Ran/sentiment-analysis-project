#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import gc

# 强制离线模式，只使用本地模型
os.environ['TRANSFORMERS_OFFLINE'] = '1'

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess():
    """加载并预处理数据"""
    logger.info("加载500000条数据进行训练...")
    
    df = pd.read_csv('Reviews_translated.csv', encoding='utf-8', nrows=500000)
    logger.info(f"原始数据: {len(df)} 条")
    
    df = df[['Score', 'Text_Translated']].copy()
    gc.collect()
    
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    df = df.dropna(subset=['Score'])
    
    conditions = [
        df['Score'] > 3.5,
        df['Score'] < 2.5
    ]
    choices = [2, 0]
    df['Label'] = np.select(conditions, choices, default=1)
    
    df['cleaned_text'] = df['Text_Translated'].astype(str).fillna('')
    df['cleaned_text'] = df['cleaned_text'].str.slice(0, 128)
    
    df = df.drop('Text_Translated', axis=1)
    gc.collect()
    
    df = df[df['cleaned_text'].str.len() > 10]
    
    logger.info(f"最终数据量: {len(df)} 条")
    logger.info("标签分布:")
    label_counts = df['Label'].value_counts().sort_index()
    for label, count in label_counts.items():
        sentiment = {0: '负面', 1: '中性', 2: '正面'}[label]
        logger.info(f"  {sentiment}({label}): {count} 条")
    
    return df[['cleaned_text', 'Label']]

def tokenize_function(examples, tokenizer):
    """分词函数 - 修复版本"""
    # 先进行分词
    tokenized = tokenizer(
        examples['cleaned_text'],
        truncation=True,
        padding=True,
        max_length=128
    )
    
    # 确保标签被正确添加
    tokenized['labels'] = examples['Label']
    
    return tokenized

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

def main():
    logger.info("=== 使用本地中文BERT模型训练 ===")
    
    try:
        # 1. 检查GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device}")
        
        # 2. 检查本地模型是否存在
        model_path = './bert-base-chinese'
        if not os.path.exists(model_path):
            logger.error(f"本地模型不存在: {model_path}")
            logger.info("请先运行: huggingface-cli download bert-base-chinese --local-dir ./bert-base-chinese")
            return
        
        logger.info(f"找到本地模型: {model_path}")
        
        # 3. 加载数据
        df = load_and_preprocess()
        
        # 4. 划分数据集
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['Label']
        )
        
        logger.info(f"训练集: {len(train_df)} 条")
        logger.info(f"验证集: {len(val_df)} 条")
        
        # 5. 从本地加载中文BERT模型
        logger.info("从本地加载中文BERT模型...")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=3
        )
        logger.info("✓ 中文BERT模型加载成功")
        
        # 6. 移动到GPU
        model.to(device)
        logger.info(f"模型已移动到: {device}")
        
        # 7. 准备数据集 - 修复版本
        logger.info("准备训练数据集...")
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        logger.info("对文本进行分词...")
        train_dataset = train_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            batch_size=10000,
            remove_columns=train_dataset.column_names  # 移除原始列，只保留分词后的列
        )
        val_dataset = val_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True, 
            batch_size=10000,
            remove_columns=val_dataset.column_names
        )
        
        # 8. 设置训练参数
        training_args = TrainingArguments(
            output_dir="./chinese_sentiment_model",
            overwrite_output_dir=True,
            learning_rate=2e-5,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=32,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_steps=5000,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to=None,
            dataloader_pin_memory=False,
        )
        
        # 9. 数据收集器
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # 10. 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        
        # 11. 开始训练
        logger.info("开始模型训练...")
        logger.info("这可能需要几个小时，请耐心等待...")
        
        train_result = trainer.train()
        
        # 12. 保存模型
        logger.info("保存训练好的模型...")
        trainer.save_model("./final_chinese_sentiment_model2.0")
        tokenizer.save_pretrained("./final_chinese_sentiment_model2.0")
        
        # 13. 最终评估
        logger.info("最终模型评估...")
        eval_results = trainer.evaluate()
        
        logger.info("=== 训练结果 ===")
        logger.info(f"训练损失: {train_result.metrics['train_loss']:.4f}")
        logger.info(f"验证准确率: {eval_results['eval_accuracy']:.4f}")
        
        logger.info("=== 训练完成! ===")
        logger.info(f"模型保存路径: ./final_chinese_sentiment_model2.0")
        
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
