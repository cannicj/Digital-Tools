from IPython.display import display, clear_output
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
        'SP500': (0.0, 0.4470588235294118, 0.6980392156862745),  # Blue
        'DTC': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  # Green
        'RFC': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # Red
        'SVC': (0.8, 0.4745098039215686, 0.6549019607843137)  # Orange
    }

    # Use the custom color palette
    for i in range(1, cols):
        col_title = dataframe.columns[i]
        ax.plot(dataframe["DATE"], dataframe.iloc[:, i], label=col_title, color=color_palette.get(col_title, 'k'))  # 'k' is black for any undefined column title

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
    else:
        plt.show()
