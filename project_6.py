import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# Sample data as multi-line string
data_str = """
SalesID,Date,ProductID,OrderQty,CustomerID,ProductName,Category,Subcategory,PriceSold
1,2023-01-01,1,2,1,Bike Helmet,Safety Equipment,Head Gear,35
2,2023-01-01,2,1,2,Mountain Bike,Bikes,Mountain Bikes,800
3,2023-01-02,3,3,3,Road Bike,Bikes,Road Bikes,600
4,2023-01-02,1,1,4,Bike Helmet,Safety Equipment,Head Gear,35
5,2023-01-03,2,1,5,Mountain Bike,Bikes,Mountain Bikes,800
6,2023-01-03,3,2,1,Road Bike,Bikes,Road Bikes,600
7,2023-01-04,1,3,2,Bike Helmet,Safety Equipment,Head Gear,35
8,2023-01-04,2,2,3,Mountain Bike,Bikes,Mountain Bikes,800
9,2023-01-05,3,1,4,Road Bike,Bikes,Road Bikes,600
10,2023-01-05,1,2,5,Bike Helmet,Safety Equipment,Head Gear,35
"""

# Convert multi-line string to pandas DataFrame
data = pd.read_csv(StringIO(data_str))

# Create a new column for revenue
data['Revenue'] = data['OrderQty'] * data['PriceSold']

# Aggregate sales data by product category
sales_by_category = data.groupby('Subcategory').agg({'Revenue': 'sum'})

# Plotting
sales_by_category.plot(kind='bar', legend=None)
plt.title('Sales Figures by Product Subcategory')
plt.xlabel('Product Category')
plt.ylabel('Sales Revenue')
plt.xticks(rotation=45)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# Sample data as multi-line string
data_str = """
SalesID,Date,ProductID,OrderQty,CustomerID,Country,ProductName,Category,Subcategory,PriceSold
1,2023-01-01,1,2,1,USA,Bike Helmet,Safety Equipment,Head Gear,35
2,2023-01-01,2,1,2,Germany,Mountain Bike,Bikes,Mountain Bikes,800
3,2023-01-02,3,3,3,USA,Road Bike,Bikes,Road Bikes,600
4,2023-01-02,1,1,4,Germany,Bike Helmet,Safety Equipment,Head Gear,35
5,2023-01-03,2,1,5,USA,Mountain Bike,Bikes,Mountain Bikes,800
6,2023-01-03,3,2,1,Germany,Road Bike,Bikes,Road Bikes,600
7,2023-01-04,1,3,2,USA,Bike Helmet,Safety Equipment,Head Gear,35
8,2023-01-04,2,2,3,Germany,Mountain Bike,Bikes,Mountain Bikes,800
9,2023-01-05,3,1,4,USA,Road Bike,Bikes,Road Bikes,600
10,2023-01-05,1,2,5,Germany,Bike Helmet,Safety Equipment,Head Gear,35
"""

# Convert multi-line string to pandas DataFrame
data = pd.read_csv(StringIO(data_str))

# Create a new column for revenue
data['Revenue'] = data['OrderQty'] * data['PriceSold']

# Aggregate sales data by subcategory and country
sales_by_subcategory_country = data.groupby(['Category', 'Country']).agg({'Revenue': 'sum'}).reset_index()

# Pivot data for comparison
sales_pivot = sales_by_subcategory_country.pivot(index='Category', columns='Country', values='Revenue')

# Plotting
sales_pivot.plot(kind='bar')
plt.title('Sales Figures by Category and Country')
plt.xlabel('Category')
plt.ylabel('Sales Revenue')

# Tilt x-axis labels
plt.xticks(rotation=45)

plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# Sample data as multi-line string
data_str = """
SalesID,Date,ProductID,OrderQty,CustomerID,Country,ProductName,Category,Subcategory,PriceSold
1,2018-01-01,1,2,1,USA,Bike Helmet,Safety Equipment,Head Gear,35
2,2018-01-01,2,1,2,Germany,Mountain Bike,Bikes,Mountain Bikes,800
3,2018-01-02,3,3,3,USA,Road Bike,Bikes,Road Bikes,600
4,2018-01-02,1,1,4,Germany,Bike Helmet,Safety Equipment,Head Gear,35
5,2018-01-03,2,1,5,USA,Mountain Bike,Bikes,Mountain Bikes,800
6,2018-01-03,3,2,1,Germany,Road Bike,Bikes,Road Bikes,600
7,2018-01-04,1,3,2,USA,Bike Helmet,Safety Equipment,Head Gear,35
8,2018-01-04,2,2,3,Germany,Mountain Bike,Bikes,Mountain Bikes,800
9,2018-01-05,3,1,4,USA,Road Bike,Bikes,Road Bikes,600
10,2018-01-05,1,2,5,Germany,Bike Helmet,Safety Equipment,Head Gear,35
"""

# Convert multi-line string to pandas DataFrame
data = pd.read_csv(StringIO(data_str))

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Aggregate sales data by date
sales_by_date = data.groupby(['Date']).agg({'OrderQty': 'sum'}).reset_index()

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(sales_by_date['Date'], sales_by_date['OrderQty'])
plt.title('Sales Quantity by Day')
plt.xlabel('Date')
plt.ylabel('Sales Quantity')

plt.show()


data_str = """
SalesID,Date,ProductID,OrderQty,CustomerID,Country,ProductName,Category,Subcategory,PriceSold
1,2018-01-01,1,2,1,USA,Bike Helmet,Safety Equipment,Head Gear,35
2,2018-01-01,2,1,2,Germany,Mountain Bike,Bikes,Mountain Bikes,800
3,2018-01-02,3,3,3,USA,Road Bike,Bikes,Road Bikes,600
4,2018-01-02,1,1,4,Germany,Bike Helmet,Safety Equipment,Head Gear,35
5,2018-01-03,2,1,5,USA,Mountain Bike,Bikes,Mountain Bikes,800
6,2018-01-03,3,2,1,Germany,Road Bike,Bikes,Road Bikes,600
7,2018-01-04,1,3,2,USA,Bike Helmet,Safety Equipment,Head Gear,35
8,2018-01-04,2,2,3,Germany,Mountain Bike,Bikes,Mountain Bikes,800
9,2018-01-05,3,1,4,USA,Road Bike,Bikes,Road Bikes,600
10,2018-01-05,1,2,5,Germany,Bike Helmet,Safety Equipment,Head Gear,35
"""

# Convert multi-line string to pandas DataFrame
data = pd.read_csv(StringIO(data_str))

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Unique product names
products = data['ProductName'].unique()

# Create a separate plot for each product
plt.figure(figsize=(10, 6))

for product in products:
    product_data = data[data['ProductName'] == product]
    product_data_by_date = product_data.groupby(['Date']).agg({'OrderQty': 'sum'}).reset_index()
    plt.plot(product_data_by_date['Date'], product_data_by_date['OrderQty'], label=product)

plt.title('Sales Quantity by Day by Product')
plt.xlabel('Date')
plt.ylabel('Sales Quantity')
plt.legend()
plt.show()