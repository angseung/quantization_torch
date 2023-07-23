import pandas as pd

# ALIGNMENT: https://jsonformatter.curiousconcept.com/

df = pd.read_excel("./Quantization Status.xlsx")
json_str = df.to_json("status.json", orient="records")
