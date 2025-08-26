from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import numpy as np
import torch 

model_name = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token':'[PAD]'})

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj","k_proj","v_proj","o_proj",]
)
model = AutoModelForCausalLM.from_pretrained(model_name,device_map='auto', torch_dtype=torch.float16)

lora_model = get_peft_model(model,lora_config)

dataset = load_dataset("json", data_files="isekai_yui_dataset.jsonl")

# 🔹 'train' 스플릿 선택
train_dataset = dataset["train"]

# 🔹 최대 토큰 길이 계산 (한 번만 실행)
token_lens = [len(tokenizer.encode(sample["input"] + " " + sample["output"], add_special_tokens=True)) for sample in train_dataset]
max_token_length = int(np.percentile(token_lens, 95))  # 상위 95% 샘플 기준
print(f"📌 설정된 max_length: {max_token_length}")

# 🔹 데이터 전처리 함수
def preprocess_function(example):
    user_input = example["input"]
    assistant_response = example["output"]

    # ✅ 학습 데이터 텍스트 포맷 설정 (QA 구조)
    text = f"사용자: {user_input}\n유이: {assistant_response}"

    # ✅ 토큰화 및 패딩 적용
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",  # ✅ 최대 길이에 맞춰 패딩
        max_length=max_token_length,
        return_tensors="pt"
    )

    return {
        "input_ids": tokenized["input_ids"][0].tolist(),
        "attention_mask": tokenized["attention_mask"][0].tolist(),
        "labels": tokenized["input_ids"][0].tolist()
    }

# 🔹 데이터셋 변환 (KeyError 방지)
tokenized_dataset = train_dataset.map(preprocess_function, batched=False, remove_columns=["input", "output"])

print("\n🎉 데이터 전처리 완료!")

training_args = TrainingArguments(
    output_dir="./yui_finetuned",
    num_train_epochs=3,  # ✅ 학습 횟수 증가
    bf16=True,
    per_device_train_batch_size=8,  # ✅ LoRA는 가벼우므로 8~16 가능
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=2e-5,  # ✅ 2e-6은 너무 작음 → 2e-5로 변경
    max_steps=2500,  # ✅ `max_stps` → `max_steps` 수정
    save_steps=500,
    logging_steps=20,
    save_total_limit=2,
)

trainer = Trainer(
    model=lora_model,  # ✅ LoRA 적용된 모델 사용!
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

trainer.model.save_pretrained("./yui_finetuned")
tokenizer.save_pretrained("./yui_finetuned")

print("파인 튜닝 완료 ㅊㅋㅊㅋ")