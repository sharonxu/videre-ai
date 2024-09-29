from openai import OpenAI

YOUR_API_KEY = "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"

query = 'Hezbollah leader Hassan Nasrallah killed in Israeli airstrike in Beirut'


messages = [
    {
        "role": "system",
        "content": (
            "You are an artificial intelligence assistant and you need to "
            "give helpful, detailed search results for a user's query."
            "Be concise and to the point."
            "Give source URLs for your information."
        ),
    },
    {
        "role": "user",
        "content": (
            query
        ),
    },
]

client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")

# chat completion without streaming
response = client.chat.completions.create(
    model="llama-3.1-sonar-small-128k-online",
    messages=messages,
)
print(response)

