import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates 

###########################################
# Problem 1
###########################################

def mcev(fun, n_samples, mu, sigma):
    samples = mu + np.sqrt(sigma) * np.random.randn(n_samples)
    function_values = np.array([fun(x) for x in samples])
    return function_values.mean()

###########################################
# Problem 2
###########################################

def plot_total_volume_histogram():
    # Load data using load_data function
    stocks, dates, closing_price, volume_traded = load_data("dow_jones.csv")
    
    # Convert dates to datetime format for filtering
    dates = pd.to_datetime(date)
    
    # Identify the last six months from the latest date
    six_months_ago = dates.max() - pd.DateOffset(months=6)
    
    # Find the index range for the last six months
    last_six_months_mask = date >= six_months_ago
    last_six_months_volumes = volume[:, last_six_months_mask]

    # Compute total volume for each stock over the last six months
    total_volumes = np.sum(last_six_months_volume, axis=1)

    # Plot histogram of total volumes with 10 bins
    plt.figure(figsize=(10, 6))
    plt.hist(total_volume, bins=10, color='skyblue', edgecolor='black')

    # Set the title and labels
    plt.title("Total Trading Volume of Each Stock Over the Last Six Months")
    plt.xlabel("Total Volume")
    plt.ylabel("Number of Stocks")

    # Save the histogram as a PNG file
    plt.savefig("total_volume_histogram.png")

    # Display the plot (optional, for verification)
    plt.show()

    

  

###########################################
# Problem 3
###########################################

df = pd.read_csv('/Users/mikeyjensen/Downloads/dow_jones.csv')

df['date'] = pd.to_datetime(df['date']) 

six_months_ago = df['date'].max() - pd.DateOffset(months=6)
filtered_df = df[df['date'] >= six_months_ago]

selected_stocks = ['IBM','INTC','MSFT']
filtered_df = filtered_df[filtered_df['stock'].isin(selected_stocks)]
pivot_df = filtered_df.pivot(index= 'date', columns='stock', values='close')

plt.figure(figsize=(10,6))

for stock in selected_stocks:
    plt.plot(pivot_df.index, pivot_df[stock], label=stock)

plt.title("Closing Prices of IBM, Intel, and Microsoft Over Six Months")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend(title="Stock")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
plt.xticks(fontsize=8.5)

###########################################
# Testing Code
###########################################
def load_data(filename):
    stocks = []
    dates = []

    closing_price = np.empty((30, 25))
    volume_traded = np.empty((30, 25))

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        curr_stock = ""
        row_idx = -1
        col_idx = 0
        for row in reader:
            if row[0] not in stocks:
                stocks.append(row[0])
            if row[1] not in dates:
                dates.append(row[1])

            if row[0] != curr_stock:
                curr_stock = row[0]
                row_idx += 1
                col_idx = 0

            closing_price[row_idx, col_idx] = float(row[2])
            volume_traded[row_idx, col_idx] = int(row[3])
            col_idx += 1

    return stocks, dates, closing_price, volume_traded


if __name__ == "__main__":
    def fun(x):
        return x ** 2


    print(mcev(fun, 100000, 0.0, 1.0))

    # stocks is a list of stock names in alphabetical order
    # dates is the list of dates contained in this dataset in order, as strings
    # closing_price is an SxD array, with element s,d holding the closing price in US dollars for stock s on date d
    # volume_traded is an SxD array, with element s,d holding the volume of trades
    #   that occured for stock s during the week ending on date d
    stocks, dates, closing_price, volume_traded = load_data("dow_jones.csv")

    print(stocks)
    print(dates)
    print(closing_price)
    print(volume_traded)

    plot_closing_prices(stocks, dates, closing_price)
    plot_total_volume_histogram(volume_traded)
