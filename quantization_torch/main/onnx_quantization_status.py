import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pandas as pd

# ALIGNMENT: https://jsonformatter.curiousconcept.com/

df = pd.read_excel("./Quantization Status.xlsx")
json_str = df.to_json("status.json", orient="records")
