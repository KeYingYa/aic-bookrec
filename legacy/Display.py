import pandas as pd
import pprint
import json


# df = pd.read_csv(r"DataLoad\inter_reevaluation.csv",encoding="ANSI")
df = pd.read_parquet("DataLoad/user_inter_matrix.parquet")
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("\n" + "=" * 60)

head=df.head(2)


print(head)
print(head.columns)
print(head.shape)