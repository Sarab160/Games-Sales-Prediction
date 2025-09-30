import pandas as pd
df=pd.read_csv("vgsales.csv")
print(df.head())

print(df.info())
print(df.duplicated())

print(df.isnull().sum())
print(df["year"].mode())
print(df.shape)
df.dropna(inplace=True)
print(df.isnull().sum())
# df["year"]=df["year"].fillna(df["year"].)
# Strip spaces and force everything to string
df["platform"] = df["platform"].astype(str).str.strip()


df["platform"] = df["platform"].replace({"2600": "Atari 2600"})

print(df["platform"].unique())

df.to_csv("gamesales.csv")
