from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ✅ 저장된 모델 로딩
model_path = "./transs"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# ✅ 사용자는 상황을 생략하고 질문만 입력 가능
question = "어떤 상황에서 '약간 이렇게 쓰긴 썼어'라고 말하게 되었나요?"
prompt = f"### Question: {question}\n### Answer:"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        temperature=0.7
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
