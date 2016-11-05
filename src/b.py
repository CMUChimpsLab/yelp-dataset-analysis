import pandas as pd
import json

def load_data(filepath):
    data = []

    # Open file and read in line by line
    with open(filepath) as file:
        for line in file:
            # Strip out trailing whitespace at the end of the line
            data.append(json.loads(line.rstrip()))

    return data

data = load_data('NVpair16.json')

business_df = pd.DataFrame.from_dict(data)

print business_df.info()
print business_df.head(1)
