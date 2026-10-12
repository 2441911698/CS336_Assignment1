import torch
import numpy as np
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# ==========================================
# 1. 准备示例数据
# ==========================================
# 实际项目中可以替换为 datasets.load_dataset("csv", data_files="...")
train_data = {
    "text": [
        "This product is amazing, completely exceeded my expectations!",
        "Terrible quality, broke on the first day.",
        "Great battery life and clear display.",
        "Worst customer service ever, completely unresponsive.",
        "I love the design and it feels very premium.",
        "Do not buy this, it is a complete waste of money."
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1: 正向, 0: 负向
}

val_data = {
    "text": [
        "Highly recommended, works like a charm!",
        "Defective out of the box, returning immediately."
    ],
    "label": [1, 0]
}

train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)

# ==========================================
# 2. 分词与数据预处理 (Tokenization)
# ==========================================
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    # truncation=True 会截断超过 max_length 的句子
    # 动态 padding 交给后面的 DataCollator，能大幅提升训练速度
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# 动态填充器：每个 batch 只 padding 到该 batch 内最长句子的长度
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ==========================================
# 3. 加载预训练模型
# ==========================================
# AutoModelForSequenceClassification 会自动在 BERT 顶层挂载一个 Linear 分类头
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# ==========================================
# 4. 定义评估指标
# ==========================================
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# ==========================================
# 5. 配置训练参数 (TrainingArguments)
# ==========================================
training_args = TrainingArguments(
    output_dir="./bert_finetuned_results",
    eval_strategy="epoch",        # 每个 epoch 评估一次
    save_strategy="epoch",        # 每个 epoch 保存一次检查点
    learning_rate=2e-5,           # 微调学习率通常设得较小 (2e-5 到 5e-5)
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=1,
    load_best_model_at_end=True,  # 训练结束加载验证集表现最好的权重
    metric_for_best_model="accuracy",
)

# ==========================================
# 6. 使用 Trainer 启动微调
# ==========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("开始微调预训练模型...")
trainer.train()

# 保存最终微调好的模型和分词器
trainer.save_model("./my_best_bert_classifier")
tokenizer.save_pretrained("./my_best_bert_classifier")
print("模型微调完成并已保存！")

# ==========================================
# 7. 模型推理预测 (Inference)
# ==========================================
test_text = "The screen clarity is fantastic and setup was easy."
inputs = tokenizer(test_text, return_tensors="pt").to(model.device)

model.eval()
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()

sentiment = "正面 (Positive)" if predicted_class == 1 else "负面 (Negative)"
print(f"\n测试文本: '{test_text}'")
print(f"预测结果: {sentiment}")