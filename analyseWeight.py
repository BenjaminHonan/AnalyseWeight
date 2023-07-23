import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def readdata(filename):
    df = pd.read_csv(filename)

    df["Time of Measurement"] = pd.to_datetime(df["Time of Measurement"])
    # Remove entries taken in the afternoon
    df = df[df['Time of Measurement'].dt.time <= pd.to_datetime("12:00:00").time()].reset_index()

    df["date"] = df["Time of Measurement"].dt.date
    # Remove duplicate date entries and only take the last one. Also group the data by the date entry
    df = df.groupby("date").last().reset_index()

    # Necessary to explicitly set data type for the conversion to array of datetime
    dates = np.array(df["date"], dtype=np.datetime64)
    weights = df["Weight(kg)"].values
    return (dates,weights)

def plot_weight_progression(dates, weights, start=None):

	# Filter the data if a date range is specified
    if start:
        start = np.datetime64(start)
        mask = dates >= start
        dates = dates[mask]
        weights = weights[mask]

    # First order polynominal i.e. straight line
    coefficients = np.polyfit(dates.astype(float), weights, 1)
    poly_function = np.poly1d(coefficients)

    # Calculate weight gain/loss per month (gradient)
    # Since the x-axis is in units of days, we convert it to months by dividing by the average days in a month
    days_in_month = 30.44  # Average number of days in a month
    weight_gain_per_month = coefficients[0] * days_in_month

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, weights, 'o', label='Data Points')
    plt.plot(dates, poly_function(dates.astype(float)), 'r-', label='Best Fit Line')
    plt.xlabel('Date')
    plt.ylabel('Weight (kg)')
    plt.title('Linear Line of Best Fit')
    plt.legend()
    plt.grid(True)

    # Set date format for the x-axis labels (display dates as 'YYYY-MM-DD')
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    # Set major locator to Weeks
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator(interval=1))

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Annotate the plot with weight gain/loss per month
    annotation = f"Weight gain/loss per month: {weight_gain_per_month:.2f} kg"
    plt.text(0.02, 0.85, annotation, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8), fontsize=12, verticalalignment='top')

    date_range = "{}_{}".format(dates[0], dates[-1])

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'weight_progression_{date_range}.png')
    plt.savefig(f'weight_progression_{date_range}.pdf')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot weight progression over time.")
    parser.add_argument('--filename', type=str, default="RenphoHealth-bphonan.csv", help="Plot data only from the given date (YYYY-MM-DD).")
    parser.add_argument('--start', type=str, help="Plot data only from the given date (YYYY-MM-DD).")

    args = parser.parse_args()

    dateweights = readdata(args.filename)
    plot_weight_progression(dateweights[0], dateweights[1], start=args.start)
