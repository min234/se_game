import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ 파인튜닝된 모델 경로
fine_tuned_model_path = "./llama"

# ✅ 토크나이저 로드 (패딩 토큰 설정)
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id  # 🚀 패딩 문제 해결

# ✅ 모델 로드 (VRAM 절약을 위한 float16 적용)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    fine_tuned_model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # ✅ VRAM 절약
    low_cpu_mem_usage=True  # ✅ 메모리 사용 최적화
).to(device)

# ✅ 모델 FP16 또는 BF16 적용 (VRAM 부족 문제 방지)
if device == "cuda":
    model = model.half()  # ✅ VRAM 절약

print("🎉 LLaMA2 대화형 테스트 시작! 'exit'을 입력하면 종료됩니다.")

# 🔹 대화 루프 (무한 반복)
while True:
    user_input = input("\n👤 [사용자]: ")

    if user_input.lower() in ["exit", "quit", "종료"]:
        print("👋 대화를 종료합니다!")
        break  # 루프 종료

    # 🔹 입력을 토크나이징 (패딩 & 길이 설정 추가)
    input_ids = tokenizer(
        user_input,
        return_tensors="pt",
        padding=True,  # ✅ max_length 없이 자동 패딩
        truncation=True,
        max_length=512  # 🚀 토큰 길이 설정
    ).input_ids.to(device)

    # 🔹 모델 예측 생성 (최적화된 설정 적용)
    output = model.generate(
        input_ids,
        max_new_tokens=100,  # ✅ 추가 생성 토큰 개수 제한
        temperature=0.7,  # 창의적인 답변을 원하면 높이고, 보수적이면 낮춤
        top_k=50,         # 상위 K개 단어 중 선택
        top_p=0.9,        # 확률 높은 단어 위주 선택
        repetition_penalty=1.2,  # 🚀 반복되는 단어 방지
        eos_token_id=tokenizer.eos_token_id  # 🚀 잘못된 반복 생성 방지
    )

    # 🔹 결과 출력 (프롬프트 제외)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # ✅ 프롬프트 제거
    if user_input in response:
        response = response.split(user_input)[-1].strip()

    print(f"\n🤖 [AI]: {response}")
