import pandas as pd

df = pd.read_excel('./data.xlsx', sheet_name=0)

columns_name = df.columns.values

all_prompts = []

for i in range((df.index.values.max()+1)):
    prompt = ""
    for j in range(2,8):
        prompt = prompt + columns_name[j] + str(df.iloc[i,j])
    all_prompts.append(prompt)

print(len(all_prompts))
df['prompt'] = all_prompts

df.to_excel('result3.xlsx', index=False)
print('success')