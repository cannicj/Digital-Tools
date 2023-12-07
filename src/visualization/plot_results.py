from IPython.display import display, clear_output
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def plot_results(dataframe, include_sp500, fig=None, plot_output_widget=None):
    # Plotting
    if fig is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        ax = fig.gca()
        ax.clear()

    cols = dataframe.shape[1]
    
    # Define a custom color palette based on column titles
    color_palette = {
        'SP500': 'blue',
        'DTC': 'green',
        'RFC': 'red',
        'SVC': 'orange'
    }

    # Use the default Seaborn color palette for colorblind viewers
    default_color_palette = sns.color_palette("colorblind")

    # Set the Seaborn color palette with custom colors for specific columns
    sns.set_palette([color_palette.get(col, default_color_palette[i]) for i, col in enumerate(dataframe.columns[1:])])

    for i in range(1, cols):
        col_title = dataframe.columns[i]
        ax.plot(dataframe["DATE"], dataframe.iloc[:, i], label=col_title, color=color_palette.get(col_title, default_color_palette[i-1]))

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.legend()
    ax.set_title("Financial Performance Prediction" + (" with " if include_sp500 else " without ") + "S&P500")
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    # Add grid
    ax.grid(True)

    # If using an output widget, clear previous output
    if plot_output_widget:
        with plot_output_widget:
            clear_output(wait=True)
    else :
        plt.show()


