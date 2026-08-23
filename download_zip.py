import urllib.request
import zipfile
import io
import os

url = "https://github.com/Orphanet/Orphadata_aggregated/archive/refs/heads/main.zip"
print(f"Downloading {url}...")

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as resp:
        content = resp.read()
        print(f"Downloaded {len(content)} bytes. Extracting...")
        z = zipfile.ZipFile(io.BytesIO(content))
        z.extractall("data/orphadata_zip")
        print("Extracted successfully!")
except Exception as e:
    print("Error:", e)
