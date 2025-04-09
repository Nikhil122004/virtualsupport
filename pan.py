import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
dataset = pd.read_csv(r'C:\Users\Lenovo\Desktop\Store.csv')


# Set the style for the plots
sns.set(style="whitegrid")

# Visualization 1: Sales vs. Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Price', y='Sales', data=dataset, hue='Category', palette='Set1')
plt.title('Sales vs. Price')
plt.xlabel('Price')
plt.ylabel('Sales')
plt.legend(title='Category')
plt.show()

# Visualization 2: Stock Quantity vs. Sales
plt.figure(figsize=(10, 6))
sns.barplot(x='Product Name', y='Stock Quantity', data=dataset, palette='viridis')
plt.xticks(rotation=90)
plt.title('Stock Quantity by Product')
plt.xlabel('Product Name')
plt.ylabel('Stock Quantity')
plt.show()

# Visualization 3: Distribution of Prices
plt.figure(figsize=(10, 6))
sns.histplot(dataset['Price'], bins=10, kde=True, color='blue')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Visualization 4: Category-wise Sales Comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Sales', data=dataset, palette='Set2')
plt.title('Category-wise Sales Distribution')
plt.xlabel('Category')
plt.ylabel('Sales')
plt.show()
