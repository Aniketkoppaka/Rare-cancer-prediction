import urllib.request
import json

req = urllib.request.Request(
    'https://api.github.com/repos/Orphanet/Orphadata_aggregated/contents',
    headers={'User-Agent': 'Mozilla/5.0'}
)
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        for item in data:
            print(f"{item['name']} ({item['type']}) - size: {item.get('size', 0)} bytes, url: {item.get('download_url')}")
except Exception as e:
    print('Error:', e)
