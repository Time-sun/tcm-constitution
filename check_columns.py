import pandas as pd
df = pd.read_csv('constitution_data.csv', encoding='utf-8-sig')
print("实际特征列名：")
for i, col in enumerate(df.columns):
    if col != 'label':
        print(f"{i}: '{col}'")