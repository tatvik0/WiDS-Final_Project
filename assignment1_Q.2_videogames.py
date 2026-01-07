import pandas as pd
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

# 1. Read the CSV directly from the Google Drive link

url = "https://drive.google.com/file/d/1AJ0mM_4D5om7lSUKFD0htM3XtKrfmp2w/view"
path = "https://drive.google.com/uc?export=download&id=" + url.split("/")[-2]

df = pd.read_csv(path)
print("First few rows of the dataset:")
print(df.head())

# Expected columns for this kind of video-game-sales dataset:
# Name, Platform, Year_of_Release or Year, Genre, Publisher,
# NA_Sales, EU_Sales, JP_Sales, Other_Sales, (maybe Global_Sales)

# (a) Add 'Global_Sales', sort (highest first), print

region_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
df["Global_Sales"] = df[region_cols].sum(axis=1)   # sum of all regional sales
df_sorted = df.sort_values("Global_Sales", ascending=False)
print("\nTop 20 games by global sales:")
print(df_sorted.head(20))


# (b) Plot: total copies sold per genre globally

genre_global = df.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
genre_global.plot(kind="bar", color="steelblue")
plt.title("Total Global Video Game Sales by Genre")
plt.xlabel("Genre")
plt.ylabel("Total Global Sales (millions)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# (c) Only games containing 'Grand Theft Auto'

gta_mask = df["Name"].str.contains("Grand Theft Auto", case=False, na=False)
gta_df = df.loc[gta_mask].copy()

# handle possible column name difference for year
year_col = "Year_of_Release" if "Year_of_Release" in gta_df.columns else "Year"

gta_df["EU_JP_Sales"] = gta_df["EU_Sales"] + gta_df["JP_Sales"]

gta_result = gta_df[["Name", "Platform", year_col, "EU_JP_Sales"]]
print("\nGrand Theft Auto games (name, platform, year, EU+JP sales):")
print(gta_result)


# (d) Pie chart of GTA regional totals

gta_totals = gta_df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum()
labels = ["North America", "Europe", "Japan", "Other"]

plt.figure(figsize=(6, 6))
plt.pie(gta_totals.values, labels=labels, autopct="%1.1f%%", startangle=90)
plt.title("Total Sales of Grand Theft Auto Games by Region")
plt.tight_layout()
plt.show()