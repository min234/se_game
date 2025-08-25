from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 🔹 LLaMA2 모델 로드
model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)

# 🔹 패딩 토큰 설정 (오류 해결 ✅)
tokenizer.pad_token = tokenizer.eos_token

# 🔹 LoRA 설정 (빠른 파인튜닝을 위해 적용)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# 🔹 모델에 LoRA 적용
model = AutoModelForCausalLM.from_pretrained(model_name, token=True)
model.resize_token_embeddings(len(tokenizer))  # 새로운 pad_token을 반영할 경우 필수

model = get_peft_model(model, lora_config)

# 🔹 JSONL 데이터셋 로드
dataset = load_dataset("json", data_files="train_data.jsonl")


# 🔹 데이터 변환 함수
def preprocess_function(example):
    user_prompt = ""
    assistant_response = ""

    # 🔹 messages 리스트에서 user와 assistant 역할 찾기
    for message in example["messages"]:
        if message["role"] == "user":
            user_prompt = message["content"]
        elif message["role"] == "assistant":
            assistant_response = message["content"]

    # 🔹 올바른 텍스트 형식으로 변환
    text = f"User: {user_prompt}\nAI: {assistant_response}"

    # 🔹 토크나이징 후 반환 (패딩 설정 추가 ✅)
    tokenized = tokenizer(text, padding=True, truncation=True, max_length=512)

    # 🔹 labels를 패딩 부분만 -100으로 변경하여 loss 계산 오류 방지
    labels = tokenized["input_ids"]
    labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels  # ✅ 패딩 토큰을 -100으로 마스킹
    }

# 🔹 `batched=False`를 유지하여 리스트 중첩 방지
tokenized_dataset = dataset.map(preprocess_function, batched=False, remove_columns=["messages"])

# 🔹 데이터 크기 설정
num_examples = 40000  # 데이터 개수
batch_size = 2  # 배치 크기 (작은 GPU면 1로 설정 가능)
gradient_accumulation_steps = 4  # 작은 배치 크기를 보완하는 누적 그래디언트
max_steps = int(num_examples / batch_size) * 2  # 데이터셋을 2번 반복 학습

print(f"🔥 최적화된 max_steps: {max_steps}")  # 예상값 확인

# 🔹 학습 설정
training_args = TrainingArguments(
    output_dir="./llama_finetuned",
    per_device_train_batch_size=batch_size,  
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=100,
    max_steps=max_steps, # 전체 데이터셋을 2번 반복 학습
    # learning_rate=2e-5,
    fp16=True,  # 16-bit 학습 (메모리 절약)
    logging_dir="./logs",
    save_steps=2000,  # 2000 스텝마다 체크포인트 저장
    save_total_limit=3,
    remove_unused_columns=False
)
print("\n📂 [디버깅] Trainer에 전달할 최종 데이터셋 첫 3개 샘플:")
print(tokenized_dataset["train"][:3]) 
print("\n📂 [디버깅] tokenized_dataset의 키들:", tokenized_dataset.keys())
# 🔹 데이터 Collator 설정
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
if "train" in tokenized_dataset:
    final_dataset = tokenized_dataset["train"]
else:
    final_dataset = tokenized_dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],  
    data_collator=data_collator
)

# 🔹 파인튜닝 실행 🚀
trainer.train()

# 🔹 모델 저장
model.save_pretrained("./llama")
tokenizer.save_pretrained("./llama")
print("🎉 LLaMA2 파인튜닝 완료! 모델이 ./llama_finetuned 에 저장됨.")
