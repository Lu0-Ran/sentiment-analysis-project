import pandas as pd
import logging
import time
import os
from transformers import MarianMTModel, MarianTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# 设置日志
log_file = "translate.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 添加控制台输出
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# 强制配置所有下载都通过国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

logging.info("已配置国内镜像，所有模型下载将通过镜像网站进行...")

# 检查必要依赖
try:
    import sentencepiece
except ImportError:
    logging.error('缺少必要依赖：sentencepiece')
    logging.info('请运行: pip install sentencepiece，安装后重新启动代码')
    exit()

# 检查CUDA可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"模型设备: {device}")

# 模型配置 - 使用镜像网站下载
model_name = "Helsinki-NLP/opus-mt-en-zh"
local_model_path = "./opus-mt-en-zh"

# 加速配置
BATCH_SIZE = 64  # 增加批量大小，根据GPU内存调整
MAX_LENGTH = 128  # 减少最大长度，加快处理速度
NUM_BEAMS = 2  # 减少束搜索数量，加速生成
USE_FP16 = True  # 使用半精度浮点数，大幅加速GPU计算

def download_model():
    """通过镜像下载模型"""
    logging.info(f"开始从镜像网站下载模型: {model_name}")
    try:
        tokenizer = MarianTokenizer.from_pretrained(
            model_name,
            cache_dir=local_model_path,
            force_download=False,
            resume_download=True
        )
        
        model = MarianMTModel.from_pretrained(
            model_name,
            cache_dir=local_model_path,
            force_download=False,
            resume_download=True,
            dtype=torch.float16 if USE_FP16 and device.type == "cuda" else torch.float32
        ).to(device)
        
        # 如果使用FP16且是CUDA设备，启用优化
        if USE_FP16 and device.type == "cuda":
            model.half()  # 转换为半精度
        
        logging.info("模型下载成功!")
        return tokenizer, model
        
    except Exception as e:
        logging.error(f"模型下载失败: {str(e)}")
        return None, None

# 自定义数据集类，用于批量处理
class TranslationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        if not isinstance(text, str):
            text = str(text)
        # 简单的文本清理
        text = text.strip()
        if len(text) == 0:
            text = " "  # 空文本用空格代替
        
        # 分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,  # 不在这里填充，在collate_fn中统一处理
            truncation=True,
            return_tensors=None
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'text': text,
            'idx': idx
        }

# 自定义collate函数，用于数据加载器
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    texts = [item['text'] for item in batch]
    indices = [item['idx'] for item in batch]
    
    # 动态填充
    padded = tokenizer.pad(
        {'input_ids': input_ids, 'attention_mask': attention_mask},
        padding=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': padded['input_ids'],
        'attention_mask': padded['attention_mask'],
        'texts': texts,
        'indices': indices
    }

# 加载模型
try:
    if os.path.exists(local_model_path):
        logging.info("尝试从本地加载模型...")
        tokenizer = MarianTokenizer.from_pretrained(local_model_path)
        model = MarianMTModel.from_pretrained(
            local_model_path,
            dtype=torch.float16 if USE_FP16 and device.type == "cuda" else torch.float32
        ).to(device)
        
        if USE_FP16 and device.type == "cuda":
            model.half()
            
        logging.info("本地模型加载成功")
    else:
        logging.info("本地未找到模型，将通过镜像网站下载...")
        tokenizer, model = download_model()
        if tokenizer is None or model is None:
            exit()
            
except Exception as e:
    logging.error(f"模型加载失败: {str(e)}")
    exit()

def translate_batch(batch):
    """批量翻译文本"""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    with torch.no_grad():  # 禁用梯度计算，减少内存使用
        # 使用更快的生成参数
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_LENGTH + 10,  # 略大于输入长度
            num_beams=NUM_BEAMS,
            early_stopping=True,
            do_sample=False,  # 不使用采样，加快速度
            num_return_sequences=1
        )
    
    # 解码所有结果
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translations

def batch_translate_fast(texts):
    """快速批量翻译 - 使用DataLoader和批处理"""
    if not texts:
        return []
    
    total = len(texts)
    logging.info(f"开始快速批量翻译，共需处理 {total} 条文本")
    
    # 创建数据集和数据加载器
    dataset = TranslationDataset(texts, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,  # 并行数据加载
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False  # 如果使用GPU，固定内存
    )
    
    # 初始化结果数组
    results = [""] * total
    
    # 使用进度条
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="翻译进度")):
        try:
            translations = translate_batch(batch)
            
            # 将结果放回正确位置
            for i, idx in enumerate(batch['indices']):
                if idx < total:
                    results[idx] = translations[i]
            
            # 每处理一定批次显示进度
            if (batch_idx + 1) % 100 == 0:
                processed = min((batch_idx + 1) * BATCH_SIZE, total)
                progress = processed / total * 100
                logging.info(f"当前进度: {processed}/{total} 条 ({progress:.1f}%)")
                
        except Exception as e:
            logging.error(f"批次 {batch_idx} 翻译失败: {str(e)}")
            # 记录失败但继续处理
            for idx in batch['indices']:
                if idx < total:
                    results[idx] = ""
    
    return results

def batch_translate_simple(texts):
    """简单批量翻译 - 用于小批量或调试"""
    if not texts:
        return []
    
    total = len(texts)
    logging.info(f"开始简单批量翻译，共需处理 {total} 条文本")
    
    results = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            text = str(text)
            
        text = text.strip()
        if len(text) == 0:
            results.append("")
            continue
            
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=MAX_LENGTH + 10,
                    num_beams=NUM_BEAMS,
                    early_stopping=True
                )
                
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(translated)
            
        except Exception as e:
            text_preview = text[:20] + "..." if len(text) > 20 else text
            logging.warning(f"单条文本翻译失败 (内容前20字: {text_preview})，错误: {str(e)}")
            results.append("")
        
        # 进度显示
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            progress = (i + 1) / total * 100
            logging.info(f"当前进度: {i+1}/{total} 条 ({progress:.1f}%)")
    
    return results

# 主程序
def main():
    input_csv = "Reviews.csv"
    output_csv = "Reviews_translated.csv"
    
    try:
        logging.info(f"开始读取文件: {input_csv}")
        
        # 读取CSV文件
        try:
            df = pd.read_csv(input_csv, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(input_csv, encoding='gbk')
                logging.info("使用GBK编码读取文件")
            except UnicodeDecodeError:
                df = pd.read_csv(input_csv, encoding='latin-1')
                logging.info("使用latin-1编码读取文件")
        
        if 'Text' not in df.columns:
            logging.error("文件中未找到 'Text' 列，请检查列名是否正确")
            logging.info(f"可用列名: {list(df.columns)}")
            exit()
        
        texts = df['Text'].fillna('').astype(str).tolist()
        logging.info(f"成功提取 'Text'列，共 {len(texts)} 条数据")
        
        # 根据数据量选择翻译方法
        if len(texts) > 1000:
            # 大数据量使用快速批量翻译
            translated_texts = batch_translate_fast(texts)
        else:
            # 小数据量使用简单批量翻译
            translated_texts = batch_translate_simple(texts)
        
        # 保存结果
        df['Text_Translated'] = translated_texts
        df.to_csv(output_csv, index=False, encoding='utf-8')
        logging.info(f"翻译结果已保存到: {output_csv}")
        logging.info("所有操作完成")
        
    except FileNotFoundError:
        logging.error(f"未找到文件: {input_csv}, 请确认文件和代码在同一文件夹")
    except Exception as e:
        logging.error(f"处理过程中出现错误: {str(e)}")

if __name__ == "__main__":
    # 安装额外依赖的提示
    try:
        from tqdm import tqdm
    except ImportError:
        logging.warning("未安装tqdm，将使用简单进度显示。如需更好体验，请运行: pip install tqdm")
    
    main()
