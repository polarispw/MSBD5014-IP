import json
import os
import matplotlib.pyplot as plt

# collect all json matrix profiles in current directory
json_dir = "Llama-2-7b-hf-0.33"
profile_logs = os.listdir(json_dir)
matrix_profiles = {}
for i in profile_logs:
    file = os.path.join(json_dir, i)
    with open(file, "r") as f:
        matrix_profiles.update(json.load(f))

print(f"total {len(matrix_profiles)} matrix profiles loaded")
print(matrix_profiles.keys())

# save as csv
import pandas as pd

df = pd.DataFrame(matrix_profiles).T
df.to_csv(f"matrix_profiles_{json_dir}.csv")
