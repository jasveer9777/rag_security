import os
from dotenv import load_dotenv
from openai import OpenAI

def main():
    load_dotenv()  # loads .env from current folder

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input="Say 'API key works' in one sentence."
    )

    print(response.output_text)

if __name__ == "__main__":
    main()
