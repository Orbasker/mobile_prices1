import pandas as pd

data = pd.read_csv("Mobile Price.csv")

print(data.head())


# Task 1
# Categorical Features Classification:

# Add the total screen resolution column
data["resolution"] = data["px_height"] * data["px_width"]

# Add the DPI_w column (assuming 'sc_w' is the screen width in inches)
data["DPI_w"] = data["px_width"] / data["sc_w"].replace(0, 1)

# Add the call_ratio column
data["call_ratio"] = data["battery_power"] / data["talk_time"].replace(0, 1)

# Convert memory from MB to GB
data["memory"] = data["memory"] / 1024

# Convert 'speed', 'screen', 'cores' into categorical series
data["speed"] = data["speed"].astype("category")
data["screen"] = data["screen"].astype("category")
data["cores"] = data["cores"].astype("category")

# Output of describe() function
print(data.describe(include="all"))


# Task 2
# count phones without camera

no_camera_count = data[(data["camera"] == 0) & (data["f_camera"] == 0)].shape[0]
print(no_camera_count)

# Average battery power for single-sim phones with high-res camera:

avg_battery_power = data[
    (data["sim"] == "single") & ((data["camera"] > 12) | (data["f_camera"] > 12))
]["battery_power"].mean()
print(avg_battery_power)

# ID and price of the most expensive phone with specific features:

filtered_phones = data[
    (data["wifi"] == 0) & (data["screen"] == "touch") & (data["mobile_wt"] > 145)
]
most_expensive_phone = filtered_phones[
    filtered_phones["price"] == filtered_phones["price"].max()
][["id", "price"]]
print(most_expensive_phone)


# Pivot table for Bluetooth percentage per generation split by RAM quartiles:
# First, we need to categorize RAM into quartiles
data["ram_quartile"] = pd.qcut(data["ram"], 4, labels=False)
if data["bluetooth"].dtype == "object":
    data["bluetooth"] = data["bluetooth"].map({"yes": 1, "no": 0})

data["bluetooth"] = data["bluetooth"].fillna(0).astype(int)


# create a pivot table
pivot_table = pd.pivot_table(
    data,
    values="bluetooth",
    index="gen",
    columns="ram_quartile",
    aggfunc=lambda x: (x.sum() / len(x)) * 100,
)
print(pivot_table)


# New DataFrame from a random sample of medium speed phones:
medium_speed_phones = data[data["speed"] == "medium"].sample(frac=0.5)
new_df = medium_speed_phones[
    [
        "id",
        "battery_power",
        "ram",
        "talk_time",
        "bluetooth",
        "cores",
        "sim",
        "memory",
        "price",
    ]
]
print(new_df)


# Maximum total talk time using 3 phones:

top_3_phones = new_df.nlargest(3, "talk_time")
max_talk_time = top_3_phones["talk_time"].sum()
phones_used = top_3_phones["id"].tolist()
print(max_talk_time, phones_used)
