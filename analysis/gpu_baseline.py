import os
from openai import OpenAI


def main() -> None:
    os.environ["OPENAI_API_KEY"] = "sk-no-key-needed"
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://7e4zolmwbqe2j8-8000.proxy.runpod.net/v1",
    )

    # 2. Send a completion request using your model
    response = client.completions.create(
        model="nreHieW/Llama-3.1-8B-Instruct",
        prompt="San Francisco is a",
        max_tokens=100,
        temperature=0.3,
    )

    print(response)


if __name__ == "__main__":
    main()
