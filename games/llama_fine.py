import re
import pandas as pd

def clean_repeated_phrases(sentences):
    """ 연속된 중복 단어 제거 """
    cleaned_sentences = []
    prev_sentence = ""

    for sentence in sentences:
        sentence = re.sub(r'(\b\w+\b)( \1)+', r'\1', sentence)  # 연속된 단어 삭제
        if sentence != prev_sentence:  # 중복된 문장 방지
            cleaned_sentences.append(sentence)
        prev_sentence = sentence

    return cleaned_sentences

def split_text_like_gpt_optimized(text):
    """
    GPT처럼 문장을 자연스럽게 분리하는 함수 (최적화 버전)
    - 중복 단어 제거
    - 너무 짧은 문장은 앞뒤 문장과 합침
    - 문장 끝나는 패턴 보완
    """

    # 1. 문장이 끝날 가능성이 높은 패턴 정의
    end_patterns = r'(했어|했어요|했네|합니다|됐다|돼|되네|다|네|요|거야|거죠|구나|군요|잖아|지|네요|라니까|맞지 않아|괜찮아요|있어 봐|같아요|해요|봤어|봤네|봐요|말이야|그러니까)'

    # 2. 문장을 시작할 가능성이 높은 감탄사 및 표현
    start_patterns = r'(?<=\s)(아|어|음|뭐야|잠깐만|진짜|이거|왜|그만|근데|그러면|그리고|하지만|그래서|아니|자|예|그러니까|일단|솔직히|그렇구나|혹시|그런데|바로)'

    # 3. 문장 끝나는 단어를 기준으로 1차 분리
    sentences = re.split(f'({end_patterns})', text)

    # 4. 문장 단위로 재구성 (이전 문장과 자연스럽게 연결)
    cleaned_sentences = []
    temp_sentence = ""

    for i in range(0, len(sentences) - 1, 2):
        new_sentence = sentences[i].strip() + " " + sentences[i + 1].strip()
        if len(new_sentence) > 5:  # 너무 짧은 단어 방지
            cleaned_sentences.append(new_sentence)
        else:
            if cleaned_sentences:  # 이전 문장과 합치기
                cleaned_sentences[-1] += " " + new_sentence
            else:
                cleaned_sentences.append(new_sentence)

    # 5. 구어체 특성을 반영하여 추가 분리 (더 자연스러운 흐름 유지)
    final_sentences = []
    merged_sentence = ""

    for sentence in cleaned_sentences:
        sub_sentences = re.split(start_patterns, sentence)
        for sub in sub_sentences:
            sub = sub.strip()
            if len(sub) > 5:  # 너무 짧은 문장 방지
                if merged_sentence:
                    final_sentences.append(merged_sentence.strip())
                merged_sentence = sub
            else:
                merged_sentence += " " + sub.strip()
        if merged_sentence:
            final_sentences.append(merged_sentence.strip())
            merged_sentence = ""

    # 6. 중복된 단어 자동 정리
    final_sentences = clean_repeated_phrases(final_sentences)

    return final_sentences

# 입력 텍스트
text = """아니 이거 뭐 머리로 박아도 안 죽어 이거는 어음 아 알았어 알았어 이게 그니까 무조건 빠르게 가는게 좋은게 아니라 너무 오버런 해도 안 돼 보니까음 너무 오버런을 해도 안 되고 으 아 이게 안 죽어 아니 이게 안 죽어 이게 안 죽어요 지했다 8시간 각이다 다들 치킨 시켜 아니 이게 뭐 굳이 엄청 빠르게 갈 필요는 없잖아 그냥 좀 좀 구질구질 하더라도 이런 데서 안전하게 가는게 맞지 않아 아 아니 이게 괜찮아요 아니 있어 봐 있어 봐 일단 감을 잡아야 될 거 아니야"""

# 실행
sentences = split_text_like_gpt_optimized(text)

# 결과 출력 (pandas 활용)
df = pd.DataFrame(sentences, columns=["문장"])
print(df)
