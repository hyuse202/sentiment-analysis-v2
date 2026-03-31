# GLM-5 API Notes

## API Information

**Provider:** Zhipu AI (智谱AI)
**Model:** GLM-4 (latest), GLM-3-Turbo
**Endpoint:** `https://open.bigmodel.cn/api/paas/v4/chat/completions`

## Authentication

```python
import os
API_KEY = os.environ.get("GLM_API_KEY")
# or hardcode (not recommended)
API_KEY = "your_api_key_here"
```

## Request Format

```python
import requests

response = requests.post(
    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "model": "glm-4",  # or "glm-3-turbo"
        "messages": [
            {"role": "user", "content": "Your prompt here"}
        ],
        "max_tokens": 10,
        "temperature": 0.1,  # Low for classification
        "top_p": 0.9
    }
)

result = response.json()
text = result["choices"][0]["message"]["content"]
```

## Rate Limits

| Model | RPM | TPM |
|-------|-----|-----|
| GLM-4 | 60 | 60000 |
| GLM-3-Turbo | 120 | 120000 |

## Pricing (approximate)

| Model | Input | Output |
|-------|-------|--------|
| GLM-4 | ~$0.002/1K tokens | ~$0.002/1K tokens |
| GLM-3-Turbo | ~$0.001/1K tokens | ~$0.001/1K tokens |

## Batch Processing Template

```python
import time
from tqdm import tqdm

def batch_label(articles, batch_size=10, delay=1):
    results = []
    
    for i in tqdm(range(0, len(articles), batch_size)):
        batch = articles[i:i+batch_size]
        
        for article in batch:
            label = label_with_glm5(
                title=article['title'],
                content=article['content'],
                api_key=API_KEY
            )
            results.append({
                'id': article['id'],
                'label': label
            })
        
        # Rate limit protection
        time.sleep(delay)
    
    return results
```

## Prompt Templates

### Vietnamese Sentiment Classification

```
Bạn là chuyên gia phân tích tài chính Việt Nam. Hãy phân loại sentiment của tin tức sau:

Tiêu đề: {title}
Nội dung: {content}

Chỉ trả lời một trong ba nhãn: POSITIVE, NEGATIVE, hoặc NEUTRAL.
Sentiment:
```

### Alternative (English fallback)

```
You are a financial sentiment analyst. Classify the following Vietnamese news:

Title: {title}
Content: {content}

Return only one label: POSITIVE, NEGATIVE, or NEUTRAL.
Sentiment:
```

## Error Handling

```python
import time

def call_with_retry(prompt, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            response = requests.post(...)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limit
                time.sleep(delay * (attempt + 1))
            else:
                raise Exception(f"API error: {response.status_code}")
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
```

## Get API Key

1. Go to: https://open.bigmodel.cn/
2. Register/Login
3. Go to API Keys section
4. Create new API key
5. Copy and save securely

## Notes

- GLM-4 is recommended for best quality
- Use low temperature (0.1-0.3) for classification tasks
- Implement retry logic for rate limits
- Batch processing recommended for large datasets
