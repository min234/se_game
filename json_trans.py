import pandas as pd
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Hugging Face DeepSeek 모델 사용
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # ⚡ 더 작은 모델은 "deepseek-ai/deepseek-llm-1.3b" 사용 가능
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,  # ✅ VRAM 절약 (16-bit 연산)
    device_map="auto"           # ✅ GPU 자동 할당
)

# ✅ 엑셀 파일 로드
file_path = "한국어_단발성_대화_데이터셋.xlsx"
df = pd.read_excel(file_path, engine="openpyxl", dtype=str)

# ✅ JSONL 데이터 저장 리스트
jsonl_data = []
count = 0  # 🔹 저장 개수 카운트 변수

print("🚀 데이터 변환 시작...")

for _, row in df.iterrows():
    prompt = row["Sentence"]  # 질문 가져오기
    query = f"너는 한국 10대 학생이야. 자연스럽고 친근한 말투로 이 질문에 답해줘:\n\n{prompt}"

    # ✅ 로그: 현재 처리 중인 질문 출력
    print(f"\n🔹 [{count+1}] 질문: {prompt}")

    # ✅ 토크나이징 (GPU로 이동)
    inputs = tokenizer(query, return_tensors="pt").to("cuda")

    # ✅ 모델 실행 (with torch.no_grad()로 메모리 최적화)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,  # ✅ 최대 생성 길이 설정
            temperature=0.7,      # ✅ 창의적 대답을 원하면 증가
            top_p=0.9,            # ✅ 높은 확률의 단어만 선택
            repetition_penalty=1.2  # ✅ 반복 방지
        )

    # ✅ 답변 디코딩 및 불필요한 텍스트 제거
    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # ✅ `<think>` 태그 제거
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # ✅ 질문 내용 제거 (응답에 포함된 경우)
    if query in response:
        response = response.replace(query, "").strip()

    # ✅ 로그: 생성된 답변 출력
    print(f"💬 정제된 답변: {response}")

    jsonl_data.append({
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    })

    count += 1  # 🔹 변환된 개수 증가
    

# ✅ 최종 JSONL 파일 저장
jsonl_file = "train_data.jsonl"
with open(jsonl_file, "w", encoding="utf-8") as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ✅ 최종 저장된 데이터 로그 출력
print("\n✅ 최종 JSONL 데이터 확인:")
for entry in jsonl_data:
    print(entry)

print(f"\n✅ 총 {count}개 데이터 변환 완료! 파일: {jsonl_file}")
