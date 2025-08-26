import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import openai

openai.api_key = ""

# Selenium을 통해 YouTube에서 특정 영역 텍스트 크롤링
def crawl_text_from_youtube(url):
    options = webdriver.ChromeOptions()
    options.add_argument("window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    time.sleep(5)  # 페이지 로딩 대기

    scroll_selector = ".overflow-y-scroll.h-\\[calc\\(100vh-170px\\)].pt-2.pb-4.px-4"
    scroll_area = driver.find_element(By.CSS_SELECTOR, scroll_selector)

    texts = []
    prev_scroll_top = -1

    while True:
        # 요소 내 텍스트 가져오기
        elements = driver.find_elements(By.CSS_SELECTOR, ".dark\\:text-white.text-2xl")
        for el in elements:
            text = el.text.strip()
            if text and text not in texts:
                texts.append(text)

        # 스크롤 내리기
        driver.execute_script("arguments[0].scrollTop += arguments[0].offsetHeight;", scroll_area)
        time.sleep(1)

        # 스크롤 끝 여부 확인
        current_scroll_top = driver.execute_script("return arguments[0].scrollTop;", scroll_area)
        if current_scroll_top == prev_scroll_top:
            break
        prev_scroll_top = current_scroll_top

    driver.quit()
    return texts

# 문장 분리 에이전트
def split_agent(input_text):
    prompt = f"""
    아래 텍스트를 자연스럽게 문장 단위로 나눠주세요.
    구어체 표현과 감탄사를 유지하면서 자연스러운 흐름을 유지하세요.

    텍스트:
    "{input_text}"

    결과(문장 목록 형식):
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=1500,
    )
    sentences = response.choices[0].message.content.strip().split("\n")
    sentences = [sentence.strip("-•. ") for sentence in sentences if sentence.strip()]
    return sentences

# 상황 분석 에이전트
def situation_agent(sentence):
    prompt = f"""
    다음 문장이 사용될 수 있는 상황을 한 줄로 간략히 설명하세요.

    문장: "{sentence}"

    상황 설명:
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=200,
    )
    situation = response.choices[0].message.content.strip()
    return situation

# JSON 형식으로 결과 구성
def formatter_agent(sentence, situation):
    return {
        "situation": situation,
        "response": sentence
    }

# 전체 프로세스 통합
def process_to_json(url):
    crawled_texts = crawl_text_from_youtube(url)
    final_results = []

    for text in crawled_texts:
        sentences = split_agent(text)
        for sentence in sentences:
            situation = situation_agent(sentence)
            formatted_json = formatter_agent(sentence, situation)
            final_results.append(formatted_json)

    return final_results

# 실제 실행 예시
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=QfdXgZoyHl8&t=3s"
    result_json = process_to_json(youtube_url)

    with open("output.json", "w", encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print(json.dumps(result_json, ensure_ascii=False, indent=2))
