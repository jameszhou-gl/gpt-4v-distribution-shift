# GPT-4V API Cost

[TOC]

## Estimate the cost

[The official price for GPT-4 Turbo](https://openai.com/pricing#language-models)

<img src="https://p.ipic.vip/7hi2wt.png" style="zoom:87%;" />

In the following, we run an example in PACS dataset, and calculate the cost as 0.00555$:

1. Vision price: 0.00255
2. Text: 0.01*0.3=0.00555

> https://platform.openai.com/docs/guides/vision#:~:text=GPT%2D4%20with%20Vision%2C%20sometimes,a%20single%20input%20modality%2C%20text.

```python
import base64
import requests

# OpenAI API Key
api_key = "YOUR_OPENAI_API_KEY"

# Function to encode the image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Path to your image
image_path = "path_to_your_image.jpg"

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
                    "text": "Question: What’s in this image?, Multiple Choices: [dog, elephant, giraffe, guitar, horse, house, person], Answer: Output only one choice from the list"
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

```



```bash
{'id': 'chatcmpl-8L4PTZ8tXgaMODkASz6TSrgnnrRoZ', 'object': 'chat.completion', 'created': 1700032531, 'model': 'gpt-4-1106-vision-preview', 'usage': {'prompt_tokens': 298, 'completion_tokens': 3, 'total_tokens': 301}, 'choices': [{'message': {'role': 'assistant', 'content': 'giraffe'}, 'finish_details': {'type': 'stop', 'stop': '<|fim_suffix|>'}, 'index': 0}]}
giraffe
```



## Estimated Cost Table

|                           Dataset                            | Num of Images | Estimated Cost (USD) |
| :----------------------------------------------------------: | :-----------: | :------------------: |
|                       ***Domainbed***                        |               |                      |
|                             PACS                             |     9991      |         55.5         |
|                             VLCS                             |     10729     |         59.6         |
|                          OfficeHome                          |     15588     |         86.5         |
|                       Terra Incognita                        |     24788     |        137.6         |
|                          DomainNet                           |    586575     |        3255.5        |
|                       Camelyon17-WILDS                       |    455954     |        2530.5        |
|                          ***CLIP***                          |               |                      |
|            [ImageNetv2](https://imagenetv2.org/)             |     30000     |        166.5         |
| [ImageNet-R](https://paperswithcode.com/dataset/imagenet-r)  |     30000     |        166.5         |
| [ImageNet Sketch](https://github.com/HaohanWang/ImageNet-Sketch) |     50000     |        277.5         |
| [ImageNet-A](https://paperswithcode.com/dataset/imagenet-a)  |     7500      |         41.6         |
|  [ObjectNet](https://paperswithcode.com/dataset/objectnet)   |     50000     |        277.5         |
|                                                              |               |        7054.8        |

