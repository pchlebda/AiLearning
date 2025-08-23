import pandas as pd

data = {'Color': ['Red', 'Blue', 'Green', 'Blue']}
df = pd.DataFrame(data)

df_encoded = pd.get_dummies(df, columns=['Color'], prefix='Color')

print(df_encoded)

data = {'Age': [23, 45, 18, 34, 67, 50, 21]}

df = pd.DataFrame(data)

bins = [0, 20, 40, 60, 100]

labels = ['0-20', '21-40', '41-60', '61+']

df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

print(df['Age_Group'])
