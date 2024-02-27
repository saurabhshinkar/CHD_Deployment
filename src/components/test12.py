import requests

# Define the URL of the API endpoint you want to request
url = 'http://127.0.0.1:5000/predictdata'

# Define the data to be sent in the request body (as a Python dictionary)
payload = {
    "id": 3388
}


# Send a POST request to the API endpoint with the data in the request body
response = requests.post(url, json=payload)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Print the response content (API data)
    print(response.json())
else:
    # Print an error message if the request was not successful
    print('Error:', response.status_code)
