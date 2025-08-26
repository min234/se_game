from transformers import AutoModelForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, TrainerCallback
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import numpy as np

# ✅ 모델 및 토크나이저 로드
model_name = "meta-llama/Llama-2-13b-hf"
# model_name = r"c:\hf\Llama-3.3-70B-Instruct"
#llama2 버전은 llamatokenzier 사용
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# ✅ pad_token 별도 추가 및 토크나이저 확장
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({
        "pad_token": "<|pad|>",
        "eos_token": "<|end_of_text|>",
        "bos_token": "<|begin_of_text|>",
        "unk_token": "<|unk|>",
    })

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True
)

# ✅ 모델 임베딩 크기 재조정
model.resize_token_embeddings(len(tokenizer))

# ✅ LoRA 구성 및 적용
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

print(model)

# ✅ 데이터 로드
dataset = load_dataset("CyberNative/Code_Vulnerability_Security_DPO")["train"]
print(dataset[0]["chosen"])

# ✅ Instruction 템플릿
instruction_text = """다음 코드를 보고 언어, 취약점 종류, 수정된 코드를 순서대로 출력해줘.
반드시 아래 형식을 지켜줘:

언어: (코드 언어)
취약점: (취약점 설명)
수정된 코드:
(취약점이 수정된 코드)"""

# ✅ 프롬프트 생성 함수
def build_prompt(example):
    code = example.get("rejected", example.get("input", "")).strip()
    prompt = f"""<|begin_of_text|>
### Instruction:
{instruction_text}

### Input:
{code}

### Response:
"""
    return prompt

# ✅ max_length 측정
print("📏 최대 토큰 길이 측정 중...")
all_lengths = []
for ex in dataset:
    prompt = build_prompt(ex)
    if prompt and ex.get("chosen"):
        full = prompt + ex["chosen"] + tokenizer.eos_token
        all_lengths.append(len(tokenizer(full)["input_ids"]))

max_length = int(np.percentile(all_lengths, 95))  # 🔥 최대값 대신 95% 길이
print(f"✅ 학습에 사용할 max_length: {max_length}")

# ✅ 전처리 함수
def format(example):
    prompt = build_prompt(example)
    output = example.get("chosen", "").strip()
    
    # full 텍스트에 eos_token은 여기서 제거
    full_text = prompt + output
    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
    prompt_len = len(prompt_ids)

    labels = [-100] * prompt_len + input_ids[prompt_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ✅ 데이터 분리 및 전처리
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"].map(format, remove_columns=dataset.column_names)
eval_dataset = split_dataset["test"].map(format, remove_columns=dataset.column_names)
print(model)
print(f"✅ train_dataset size: {len(train_dataset)}")
print(f"✅ eval_dataset size: {len(eval_dataset)}")

# ✅ 샘플 확인
sample = train_dataset[0]
decoded_input = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
decoded_label = tokenizer.decode(
    [t for t in sample["labels"] if t != -100],
    skip_special_tokens=True
)

print("\n📥 [디코딩된 Input]:\n")
print(decoded_input)
print("\n📤 [디코딩된 Label]:\n")
print(decoded_label)

# ✅ 커스텀 로거
class CustomLoggerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        loss = kwargs.get("logs", {}).get("loss", None)
        if loss is not None:
            print(f"🔄 Step {state.global_step}/{state.max_steps} | Loss: {loss:.4f} | Epoch: {state.epoch:.2f}")

# ✅ 학습 설정
training_args = TrainingArguments(
    output_dir="./secu_13b",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    save_strategy="steps",
    bf16=True,
    gradient_checkpointing=True
)

# ✅ Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[CustomLoggerCallback()]
)

# ✅ 학습 시작
print("✅ 학습 시작")
trainer.train()

# ✅ 모델 저장
trainer.model.save_pretrained("./securitys_13b")
tokenizer.save_pretrained("./securitys_13b")
print("✅ 저장 완료")
