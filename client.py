import requests

# API endpoint URL
url = "http://127.0.0.1:8000/predict/"

# The input text for prediction
data = {
    "input_text": "The quick brown fox"
}

# Send POST request to FastAPI server
response = requests.post(url, json=data)

# Print the full response to debug
print(f"Response status code: {response.status_code}")
print(f"Response content: {response.text}")

# Handle the response from the server
if response.status_code == 200:
    result = response.json()
    if 'input' in result and 'prediction' in result:
        print(f"Input: {result['input']}")
        print(f"Prediction: {result['prediction']}")
    else:
        print(f"Unexpected response format: {result}")
else:
    print(f"Error: {response.status_code}, {response.text}")
