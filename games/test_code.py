from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 모델 경로
model_path = "./securitys"

# 모델과 토크나이저 로드

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

# Instruction 템플릿
instruction_text = """다음 코드를 보고 언어, 취약점 종류, 수정된 코드를 순서대로 출력해줘.
반드시 아래 형식을 지켜줘:

언어: (코드 언어)
취약점: (취약점 설명)
수정된 코드:
(취약점이 수정된 코드)"""

# 테스트 코드 입력
test_code = """
import os

def run_command(cmd):
    os.system("ping " + cmd)

user_input = input("Enter hostname or IP: ")
run_command(user_input)
"""

# 프롬프트 생성
prompt = f"""<|begin_of_text|>
### Instruction:
{instruction_text}

### Input:
{test_code}

### Response:
"""

# 토크나이즈
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 생성
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# 결과 디코딩
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
response_part = decoded.split("### Response:")[-1].strip()

print("\n📤 [모델 응답 결과]")
print(response_part)
