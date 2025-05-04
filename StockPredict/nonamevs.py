import matplotlib.pyplot as plt
import pandas as pd


# Read the MA CSV file from the current directory
df_ma = pd.read_csv('Reliance_MovingAverage.csv')

# Display the first few rows of the DataFrame
print(df_ma.head())


# Read the OBV CSV file from the current directory
df_v = pd.read_csv('Reliance_On_Balence_volume.csv')

# Display the first few rows of the DataFrame
print(df_v.head())
# Step 2: Merge the DataFrames on column 'A'
merged_df = pd.merge(df_ma, df_v, on='Close', how='inner')  # or 'left', 'right', 'outer' based on how you want to merge

print(merged_df.head())
# Step 3: Plot column 'OBV' vs 'MA' based on 'ClosePrice'

plt.figure(figsize=(14, 7))
plt.plot(merged_df.index, merged_df['OBV'], label='On-Balence Volume', color='blue')
plt.plot(merged_df.index, merged_df['21-Day MA'], label='21-Day MA', color='red', linestyle='--')
plt.plot(merged_df.index, merged_df['200-Day MA'], label='200-Day MA', color='green', linestyle='--')
plt.title('Reliance Industries Stock Volume and Moving Averages', fontsize=14)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()