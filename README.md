# -LB-Chinese-Poetry-Sentiment-
This is a sentiment analysis model fine-tuned from 'qixun/bert-chinese-poem' for analyzing sentiments in traditional Chinese poems. The model can identify 7 different categories.
language: zh tags:

BERT
sentiment-analysis
chinese-poetry
text-classification
classical-chinese license: gpl-3.0 metrics:
accuracy
f1
LB Chinese Poetry Sentiment Analysis BERT
永远怀念我最亲爱的小猫蓝宝，我们都爱你

📖 模型简介 Model Description
这是一个基于 'qixun/bert-chinese-poem' 微调的中文古诗情感分析模型，专门用于分析古典诗词中的情感倾向。模型可以识别7种不同的情感类别，帮助理解诗词的情感内涵。 模型可以识别以下7种情感:积极 Positive 消极 Negative 平静 Calm 敬畏 Awe 新奇 思念 Nostalgia 向往 Aspiration

This is a sentiment analysis model fine-tuned from 'qixun/bert-chinese-poem' for analyzing sentiments in traditional Chinese poems. The model can identify 7 different categories.

🚀 快速开始 Quick Start
安装依赖 Installation
pip install transformers torch


1.基础使用 Basic Usage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


model_name = "Buddhafate/LB-Chinese-Poetry-Sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 中文情感标签 
emotion_labels = ["积极", "消极", "平静", "敬畏", "新奇", "思念", "向往"]

# 完整分析函数 
def analyze_poem_sentiment(poem):
    """分析中文诗词的主要情感类别及概率分布"""
    inputs = tokenizer(poem, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

    predicted_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_idx].item()
    return {
        "原诗": poem,
        "主要情感": emotion_labels[predicted_idx],
        "置信度": confidence,
        "情感分布": {
            emotion_labels[i]: float(prob)
            for i, prob in enumerate(probs[0])
        }
    }

# 示例诗句 
poem = """庭前病桧自萧疏，门外惊鸥不可呼。
饱听江声十年事，来寻陈迹一篇无。
投荒坐惜人将老，望鲁空嗟道巳孤。
赖有胜天坚念在，稍分肝胆与枝梧。"""

# 输出 
result = analyze_poem_sentiment(poem)

print("=" * 60)
print(" 诗词情感分析")
print("=" * 60)
print(f"\n📜 原诗：\n{result['原诗']}\n")
print(f"主要情感：{result['主要情感']}  (置信度: {result['置信度']:.2%})")

print("\n情感概率分布：")
for label, prob in sorted(result["情感分布"].items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(prob * 20)
    print(f"  {label:>4s}: {bar:<20s} {prob:>6.2%}")





2.批量处理 Batch Processing

def batch_analyze(poems):
    """
    多首诗词
    """

    if isinstance(poems, str):
        poems = [poems]
    elif not isinstance(poems, list):
        raise ValueError("输入必须是字符串或字符串列表")

    inputs = tokenizer(poems, return_tensors="pt", padding=True, truncation=True, max_length=128)
    

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_classes = torch.argmax(probs, dim=-1)
 
    results = []
    for i, (poem, pred_idx) in enumerate(zip(poems, predicted_classes)):
        pred_idx = pred_idx.item()
        confidence = probs[i][pred_idx].item()
        all_probs = {
            emotion_labels[j]: probs[i][j].item()
            for j in range(len(emotion_labels))
        }
        results.append({
            "原诗": poem,
            "主要情感": emotion_labels[pred_idx],
            "置信度": confidence,
            "情感分布": all_probs
        })

    return results



性能指标 Performance Metrics
  分类报告 Classification Report
  模型在测试集（1,609首诗词）上的表现：
  情感类别PrecisionRecallF1-ScoreSupport积极0.960.980.97297消极0.980.950.97244平静0.980.940.96249敬畏0.990.940.97212新奇0.970.940.95250思念0.960.840.8961向往0.980.900.93296
  总体指标 Overall Metrics

  Macro Average 宏平均

  Precision: 0.97
  Recall: 0.93
  F1-Score: 0.95



性能分析 Performance Analysis

  最佳表现类别：

  敬畏 (F1: 0.97, Precision: 0.99)
  积极 (F1: 0.97, Recall: 0.98)
  消极 (F1: 0.97, Precision: 0.98)


需要注意的类别：

  思念类别的样本较少（仅61个），F1分数相对较低（0.89）
  但其Precision仍达到0.96，说明模型判断为"思念"时准确率很高


整体表现：

  宏平均F1分数达到0.95，表明模型在各类别上表现均衡
  高Precision（0.97）说明模型的预测结果可信度高



技术细节 Technical Details

  基础模型 Base Model: qixun/bert-chinese-poem
  模型架构 Architecture: BERT for Sequence Classification
  训练样本 Training Samples: ~10,000 古诗词
  最大长度 Max Length: 512 tokens
  训练轮数 Epochs: 5
  批次大小 Batch Size: 24
  学习率 Learning Rate: 2e-5
  优化器 Optimizer: AdamW
  训练设备: GPU with CUDA


使用限制 Limitations

  模型主要针对古典诗词训练，对现代诗歌效果可能有限
  输入文本建议不超过512 个token
  对于含有复杂隐喻的诗句，可能需要结合上下文理解
  由于继承自GPL-3.0许可证，使用本模型的项目也需要开源



致谢 Acknowledgments

基础模型提供者: qixun/bert-chinese-poem
训练框架: Hugging Face Transformers
特别纪念: 我的爱猫蓝宝弟弟 (2016-2025)

📄 许可证 License
本模型继承自原始模型的 GPL-3.0 许可证。使用本模型的项目需要遵守相同的开源协议。
This model inherits the GPL-3.0 license from the original model. Projects using this model must comply with the same open-source license.
📮 联系方式 Contact
email：x130319@gmail.com  WeChat:NarayanaJupiter
