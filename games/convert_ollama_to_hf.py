import json
import os
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

def convert_ollama_to_hf(ollama_model_path, output_dir):
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # TinyLlama 토크나이저 사용 (오픈 소스)
    tokenizer = LlamaTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.save_pretrained(output_dir)
    
    # 모델 구성 파일 생성
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "torch_dtype": "float16",
        "transformers_version": "4.31.0",
        # Llama2 설정
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32000,
        "rope_scaling": None,
        "rms_norm_eps": 1e-5,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Ollama 모델 가중치를 PyTorch 형식으로 변환
    state_dict = torch.load(ollama_model_path, map_location="cpu")
    
    # 모델 저장
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    
    print(f"모델이 성공적으로 변환되어 {output_dir}에 저장되었습니다.")

# 변환 실행
ollama_path = r"C:\Users\min\.ollama\models\blobs\sha256-8934d96d3f08982e95922b2b7a2c626a1fe873d7c3b06e8e56d7bc0a1fef9246"
output_dir = "./converted_llama_model"
convert_ollama_to_hf(ollama_path, output_dir) 