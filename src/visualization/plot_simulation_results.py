import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def plot_simulation_results(svm_results, rf_results, dt_results):
 

    # Calculate rf quantiles
    series = rf_results.iloc[-1:,1:].iloc[0]
    median_column = series.sub(series.median()).abs().idxmin()
    lq_column = series.sub(series.quantile(0.75)).abs().idxmin()
    hq_column = series.sub(series.quantile(0.25)).abs().idxmin()
    rf_ranges = rf_results[['DATE',lq_column, median_column, hq_column]]
    rf_ranges.rename(columns={lq_column:'25% quantile',median_column:'median',hq_column:'75% quantile'},inplace=True)
    
     # Calculate dt quantiles
    series = dt_results.iloc[-1:,1:].iloc[0]
    median_column = series.sub(series.median()).abs().idxmin()
    lq_column = series.sub(series.quantile(0.75)).abs().idxmin()
    hq_column = series.sub(series.quantile(0.25)).abs().idxmin()
    dt_ranges = dt_results[['DATE',lq_column, median_column, hq_column]]
    dt_ranges.rename(columns={lq_column:'25% quantile',median_column:'median',hq_column:'75% quantile'},inplace=True)
    
    
    # Plot for SVM simulations
    plt.figure(figsize=(14, 6))
    cols = svm_results.shape[1]
    # Set the Seaborn color palette
    sns.set_palette("colorblind")
    plt.plot(svm_results["DATE"], svm_results.iloc[:,2],label='SVM',color='blue',linewidth=2)
    plt.plot(svm_results["DATE"], svm_results["SP500"], label='SP500', color='green', linewidth=2)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.legend()
    plt.title("SVM Financial Performance Prediction" + (" with " if include_sp500 else " without ") + "S&P500")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    # Add grid
    plt.grid(True)
    plt.show()

    # Plot for Random Forest simulations
    plt.figure(figsize=(14, 6))
    cols = rf_results.shape[1]
    # Set the Seaborn color palette
    sns.set_palette("colorblind")
    for i in range(2, cols):
        plt.plot(rf_results["DATE"], rf_results.iloc[:,i],color='grey', alpha=0.4)
    plt.plot(rf_results["DATE"], svm_results["SP500"], label='SP500', color='green', linewidth=2)
    plt.plot(rf_results["DATE"], rf_ranges['25% quantile'], label='25% Quantile', color='blue', linewidth=2)
    plt.plot(rf_results["DATE"], rf_ranges['median'], label='Median', color='red', linewidth=2)
    plt.plot(rf_results["DATE"], rf_ranges['75% quantile'], label='75% Quantile', color='blue', linewidth=2)
    plt.fill_between(rf_results["DATE"], rf_ranges['25% quantile'], rf_ranges['75% quantile'], color='green', alpha=0.2)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.legend()
    plt.title("Random Forest Financial Performance Prediction" + (" with " if include_sp500 else " without ") + "S&P500")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    # Add grid
    plt.grid(True)
    plt.show()

    # Plot for Decision Tree simulations
    plt.figure(figsize=(14, 6))
    cols = dt_results.shape[1]
    # Set the Seaborn color palette
    sns.set_palette("colorblind")
    for i in range(2, cols):
        plt.plot(dt_results["DATE"], dt_results.iloc[:,i],color='grey', alpha=0.4)
    plt.plot(dt_results["DATE"], dt_results["SP500"], label='SP500', color='green', linewidth=2)
    plt.plot(dt_results["DATE"], dt_ranges['25% quantile'], label='25% Quantile', color='blue', linewidth=2)
    plt.plot(dt_results["DATE"], dt_ranges['75% quantile'], label='75% Quantile', color='blue', linewidth=2)
    plt.fill_between(dt_results["DATE"], dt_ranges['25% quantile'], dt_ranges['75% quantile'], color='green', alpha=0.2)
    plt.plot(dt_results["DATE"], dt_ranges['median'], label='Median', color='red', linewidth=2)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.legend()
    plt.title("Decision Tree Financial Performance Prediction" + (" with " if include_sp500 else " without ") + "S&P500")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    # Add grid
    plt.grid(True)
    plt.show()

