import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv(r"C:\Users\Public\Documents\retail_sales_dataset.csv")
cl = data.isna().any()
print(cl)

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Grouping dataframe by product category
clothing_data = data[data['Product Category'] == 'Clothing']
beauty_data = data[data['Product Category'] == 'Beauty']
electronics_data = data[data['Product Category'] == 'Electronics']

# Calculate statistics for clothing
mean_clothing = clothing_data['Total Amount'].mean()
median_clothing = clothing_data['Total Amount'].median()
mode_clothing = clothing_data['Total Amount'].mode()[0] if not clothing_data['Total Amount'].mode().empty else None
std_clothing = clothing_data['Total Amount'].std()

# Calculate statistics for beauty
mean_beauty = beauty_data['Total Amount'].mean()
median_beauty = beauty_data['Total Amount'].median()
mode_beauty = beauty_data['Total Amount'].mode()[0] if not beauty_data['Total Amount'].mode().empty else None
std_beauty = beauty_data['Total Amount'].std()

# Calculate statistics for electronics
mean_electronics = electronics_data['Total Amount'].mean()
median_electronics = electronics_data['Total Amount'].median()
mode_electronics = electronics_data['Total Amount'].mode()[0] if not electronics_data['Total Amount'].mode().empty else None
std_electronics = electronics_data['Total Amount'].std()

# Print the results
print(f"Clothing - Mean: {mean_clothing}, Median: {median_clothing}, Mode: {mode_clothing}, Std: {std_clothing}")
print(f"Beauty - Mean: {mean_beauty}, Median: {median_beauty}, Mode: {mode_beauty}, Std: {std_beauty}")
print(f"Electronics - Mean: {mean_electronics}, Median: {median_electronics}, Mode: {mode_electronics}, Std: {std_electronics}")

# Set 'Date' column as index
clothing_data.set_index('Date', inplace=True)
beauty_data.set_index('Date', inplace=True)
electronics_data.set_index('Date', inplace=True)

# Resample data to monthly frequency and sum the total amounts
clothing_monthly = clothing_data['Total Amount'].resample('ME').sum()
beauty_monthly = beauty_data['Total Amount'].resample('ME').sum()
electronics_monthly = electronics_data['Total Amount'].resample('ME').sum()

# Plot time series data
plt.figure(figsize=(14, 7))

plt.subplot(3, 1, 1)
plt.plot(clothing_monthly, label='Clothing Sales', color='blue')
plt.title('Monthly Sales for Clothing')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(beauty_monthly, label='Beauty Sales', color='green')
plt.title('Monthly Sales for Beauty')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(electronics_monthly, label='Electronics Sales', color='red')
plt.title('Monthly Sales for Electronics')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()

plt.tight_layout()
plt.show()

#Customer and Product analysis
gender_product_analysis = data.groupby(['Gender', 'Product Category']).size().reset_index(name='Count')
print(gender_product_analysis)
plt.figure(figsize=(12, 6))
sns.barplot(x='Product Category', y='Count', hue='Gender', data=gender_product_analysis, palette='viridis')

# Customize the plot
plt.title('Customer and Product Analysis by Gender and Product Category')
plt.xlabel('Product Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Gender')

# Show the plot
plt.tight_layout()
plt.show()

# Create a pivot table for the heatmap
heatmap_data = data.pivot_table(
    values='Total Amount',
    index=pd.Grouper(key='Date', freq='ME'),
    columns=['Gender', 'Product Category'],
    aggfunc='sum',
    fill_value=0
)

# Plot the heatmap
plt.figure(figsize=(14, 7))
sns.heatmap(heatmap_data, cmap='viridis', linewidths=0.5)

# Customize the plot
plt.title('Monthly Sales Amount by Gender and Product Category')
plt.xlabel('Gender and Product Category')
plt.ylabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
