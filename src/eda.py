import pandas as pd

df = pd.read_csv("data/appointments.csv")

print(df["No-show"].value_counts())
print(df["No-show"].value_counts(normalize=True))
