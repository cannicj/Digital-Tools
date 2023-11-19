import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
def plot_results(dataframe, include_sp500):
    # Plotting
    plt.figure(figsize=(14, 6))
    cols = dataframe.shape[1]
    # Set the Seaborn color palette
    sns.set_palette("colorblind")
    for i in range(1, cols):
     plt.plot(dataframe["DATE"], dataframe.iloc[:,i], label= dataframe.columns[i])

    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.legend()
    plt.title("Financial Performance Prediction" + (" with " if include_sp500 else " without ") + "S&P500")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.savefig('classifier.svg', dpi=1000)
    plt.show()
