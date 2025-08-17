# pip install requests
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

prompt_template = """
[Role]:
[Intent]:
[Context]:
[Instruction]: 
[Style & Constraints]: 
[Format]: 
"""

question = "Explain how AI works in a few words"

full_prompt = f"{prompt_template}\n\n[Question]: {question}"

payload = {
    "model": "gemma3:1b",  # mistral
    "prompt": full_prompt,
    "stream": False  
}

# Make the request to Ollama
resp = requests.post(OLLAMA_URL, data=json.dumps(payload), headers={"Content-Type": "application/json"})
resp.raise_for_status()
data = resp.json()

print(data["response"])
