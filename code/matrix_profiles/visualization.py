import json
import os
import matplotlib.pyplot as plt

# collect all json matrix profiles in current directory
profile_logs = os.listdir("./")
profile_logs = [i for i in profile_logs if i.endswith(".json") and "0.5" in i]
matrix_profiles = {}
for i in profile_logs:
    with open(i, "r") as f:
        matrix_profiles.update(json.load(f))

print(f"total {len(matrix_profiles)} matrix profiles loaded")
print(matrix_profiles.keys())

# save as csv
import pandas as pd

df = pd.DataFrame(matrix_profiles).T
df.to_csv("matrix_profiles.csv")
