from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# ✅ 모델 이름
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# ✅ 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # padding 시 EOS로 채움

# ✅ 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # GPU 자동 할당
    torch_dtype=torch.bfloat16,  # 또는 float16 (Windows에서)
    load_in_4bit=True
)

# ✅ LoRA 구성
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# ✅ LoRA 적용
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# ✅ 데이터 로드
dataset = load_dataset("json", data_files="memo.jsonl", split="train")

# ✅ 프롬프트 템플릿 구성
prompt_template = """### Instruction: {}\n### Question: {}\n### Answer: {}"""

# ✅ 최대 길이 자동 계산 (데이터 기반 max_length 설정)
def build_prompt(example):
    return prompt_template.format(example["instruction"], example["input"], example["output"])

token_lens = [len(tokenizer(build_prompt(e))["input_ids"]) for e in dataset]
max_length = max(token_lens)
print(f"✅ 최대 토큰 길이 (max_length): {max_length}")

# ✅ 데이터 전처리 함수
def format(example):
    prompt = build_prompt(example)
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    return {
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0],
        "labels": tokenized["input_ids"][0]
    }

# ✅ train / eval 나누기 (90:10)
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"].map(format, batched=False, remove_columns=["lang", "vulnerability", "system","question","chosen","rejected"])
eval_dataset  = split_dataset["test"].map(format, batched=False, remove_columns=["lang", "vulnerability", "system","question","chosen","rejected"])
from transformers import TrainerCallback

class CustomLoggerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        loss = logs.get("loss", None)

        if loss is not None:
            print(f"🔄 Step {state.global_step}/{state.max_steps} | Loss: {loss:.4f} | Epoch: {state.epoch:.2f}")
# ✅ 학습 설정
training_args = TrainingArguments(
    output_dir="./trans",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    num_train_epochs=20,
    logging_dir="./logs",
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    # ✅ 평가 및 저장
    evaluation_strategy="steps",   # 매 step마다 평가
    eval_steps=500,                # 500 step마다 평가
    load_best_model_at_end=True,   # 가장 낮은 loss 기준으로 모델 저장
    metric_for_best_model="loss",
    greater_is_better=False,

    save_strategy="steps",         # step마다 저장

    # ✅ mixed precision
    fp16=True,
)

# ✅ Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # ✅ 평가용 데이터 꼭 넣기
    callbacks=[CustomLoggerCallback()] 
)

# ✅ 학습 시작
print("✅ 학습 시작")
trainer.train()

# ✅ 모델 저장
trainer.model.save_pretrained("./transs")
tokenizer.save_pretrained("./transs")
print("✅ 저장 완료")
