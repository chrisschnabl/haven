from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-xxx")
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "<image_url>"},
                },
                {"type": "text", "text": "What does the image say"},
            ],
        }
    ],
)
print(response)