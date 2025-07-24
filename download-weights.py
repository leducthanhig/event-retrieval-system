import subprocess
from os.path import join, dirname
import os
import requests

package = 'transnetv2'
try:
    # Run the `pip show` command and capture the output
    result = subprocess.run(
        ["pip", "show", package],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    # Extract the `Location` field
    for line in result.stdout.splitlines():
        if line.startswith("Location:"):
            location = line.split(":", 1)[1].strip()
            print(f"Installation directory: {location}")
            break
    else:
        raise Warning(f"Location field not found for package {package}.")
except subprocess.CalledProcessError:
    raise Warning(f"Failed to retrieve information for package {package}.")

# Download weights-related files manually
urls = [
    "https://media.githubusercontent.com/media/soCzech/TransNetV2/refs/heads/master/inference/transnetv2-weights/variables/variables.data-00000-of-00001?download=true",
    "https://media.githubusercontent.com/media/soCzech/TransNetV2/refs/heads/master/inference/transnetv2-weights/variables/variables.index?download=true",
    "https://media.githubusercontent.com/media/soCzech/TransNetV2/refs/heads/master/inference/transnetv2-weights/saved_model.pb?download=true"
]

files = [
    "variables/variables.data-00000-of-00001",
    "variables/variables.index",
    "saved_model.pb"
]

for url, file in zip(urls, files):
    file_path = join(location, 'transnetv2/transnetv2-weights', file)
    os.makedirs(dirname(file_path), exist_ok=True)  # Ensure the directory exists

    print(f"Downloading {url} to {file_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for HTTP issues

    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):  # Download in chunks
            f.write(chunk)

    print(f"Downloaded {file} successfully.")
