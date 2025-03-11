from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和 tokenizer
model_path = "./finetuned_models/final_save/"#'./model/deepseek-sft/1.5b/'#./finetuned_models/final_save/
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 禁用量化或尝试使用 8-bit 量化
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # 使用 8-bit 量化
# quantization_config = BitsAndBytesConfig(load_in_4bit=False)  # 或者禁用量化
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, device_map="auto")
print("量化模型加载完毕")


from transformers import pipeline
pipe = pipeline("text-generation",model = model,tokenizer=tokenizer)
prompt = "成分过冷在化学气相沉积中的应用"

generated_texts = pipe(prompt,max_length=2096,num_return_sequences=1)


print("开始回答问题：",generated_texts[0]["generated_text"])