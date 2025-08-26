import pandas as pd
import json

# 🔹 CSV 파일 불러오기
csv_file = "isekai_yui_large_dataset.csv"
jsonl_file = "isekai_yui_dataset.jsonl"  # 변환된 JSONL 파일 이름

# 🔹 CSV 데이터 읽기
df = pd.read_csv(csv_file)

# 🔹 JSONL 형식으로 변환 후 저장
with open(jsonl_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        json_record = {
            "input": row["input"],  # CSV의 input 열
            "output": row["output"]  # CSV의 output 열
        }
        f.write(json.dumps(json_record, ensure_ascii=False) + "\n")

print(f"✅ JSONL 파일 변환 완료! ({jsonl_file})")
