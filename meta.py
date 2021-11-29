import json
import os


with open('data/physionet_meta.json') as f:
    d = json.load(f)

d2 = {}
for path, entry in d.items():
    print(os.path.split(path))
    entry['path'] = "/".join(["data/physionet"] + path.split('/')[-2:])
    d2[entry['path']] = entry

with open('data/physionet_meta_relative.json', 'w') as f:
    json.dump(d2, f, indent=2)

