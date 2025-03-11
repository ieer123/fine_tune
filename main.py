from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和 tokenizer
model_path = './model/deepseek-sft/1.5b/'
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 禁用量化或尝试使用 8-bit 量化
from transformers import BitsAndBytesConfig
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # 使用 8-bit 量化
# quantization_config = BitsAndBytesConfig(load_in_4bit=False)  # 或者禁用量化
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda")#(model_path, quantization_config=quantization_config, device_map="cuda")
print("量化模型加载完毕")

from peft import get_peft_model, LoraConfig, TaskType

# LoRA配置
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
print("LoRA加载完成")

from data_gen import samples
import json

# 准备数据
with open('material_data.jsonl', 'w', encoding='utf-8') as f:
    for s in samples:
        json_line = json.dumps(s, ensure_ascii=False)
        f.write(json_line + "\n")
    else:
        print("准备数据完毕")

from datasets import load_dataset
dataset = load_dataset(path="json", data_files={"train": "material_data.jsonl"}, split="train")
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
print(len(train_dataset), len(eval_dataset))

# Tokenizer 函数
def tokenizer_function(many_samples):
    texts = [f"{prompt}\n{completion}" for prompt, completion in zip(many_samples["instruction"], many_samples["output"])]
    tokens = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_train_dataset = train_dataset.map(tokenizer_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenizer_function, batched=True)

print("完成 tokenizing")

from transformers import TrainingArguments, Trainer

# 训练参数设置，降低批量大小和增加梯度累积
training_args = TrainingArguments(
    output_dir='./finetuned_models',
    num_train_epochs=30,
    per_device_train_batch_size=4,  # 降低批量大小
    per_device_eval_batch_size=8,  # 降低评估批量大小
    gradient_accumulation_steps=4,  # 增加梯度累积步数
    fp16=True,  # 使用混合精度训练
    logging_steps=10,
    save_steps=100,
    eval_steps=10,
    learning_rate=3e-5,
    logging_dir="./finetuned_models/logs",
    run_name="deepseek-r1-distill-finetune"
)
print("训练参数设置完毕")

# 创建 Trainer 对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset
)

# 训练
print("开始训练")
trainer.train()
print("训练完成")


save_path = "./finetuned_models/saved_models"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("---Lora模型已保存---")

#保存权重
final_save_path = "./finetuned_models/final_save"
print("---开始保存---")
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(model_path)
model = PeftModel.from_pretrained(base_model,save_path)
model = model.merge_and_unload()

model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)

print("---全量模型已经保存---")

from transformers import pipeline
pipe = pipeline("text-generation",model = model,tokenizer=tokenizer)
prompt = "化学气相沉积（CVD）是什么？"

generated_texts = pipe(prompt,max_length=512,num_return_sequences=1)


print("开始回答问题：",generated_texts[0]["generated_text"])