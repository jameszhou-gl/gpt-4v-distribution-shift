import base64
import requests

# OpenAI API Key
api_key = "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"

# Function to encode the image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Path to your image
image_path = "/home/guanglin.zhou/code/gpt-4v-distribution-shift/sketch/giraffe/n02439033_11328-1.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Question: What’s in this image?; Multiple Choices: [dog, elephant, giraffe, guitar, horse, house, person]; Output a single choice from the choices list, the normalized prediction confidence score for each choice and reason"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 300
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())
# print(response)
print(response.json()['choices'][0]['message']['content'])
