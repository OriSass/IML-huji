import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from polynomial_fitting import PolynomialFitting

PLOT_DIR = "../plots"
RANDOM_SEED = 40
MINIMAL_K = 5


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # Load CSV with proper parsing of 'Date' column
    df = pd.read_csv(filename, parse_dates=["Date"])

    df = df[df["Temp"] > -50]

    # Add 'DayOfYear' column
    df['DayOfYear'] = df['Date'].dt.dayofyear

    return df


def plot_temp_by_day(df: pd.DataFrame):
    """
    Create a scatter plot of daily temperature vs. day of year,
    color-coded by year (discrete scale).
    """
    # Set a clean style
    sns.set(style="whitegrid")

    # Create scatter plot
    plt.figure(figsize=(12, 6))
    scatter = sns.scatterplot(
        data=df,
        x="DayOfYear",
        y="Temp",
        hue="Year",  # Discrete color mapping
        palette="tab10",  # Use a discrete colormap (tab10, Set1, etc.)
        linewidth=0,
        s=20  # Dot size
    )

    # Customize plot
    plt.title("Daily Temperature vs. Day of Year by Year")
    plt.xlabel("Day of Year")
    plt.ylabel("Temperature (°C)")
    plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/temp_vs_day.png")
    # Show plot
    # plt.show()


def plot_monthly_temp_std(df: pd.DataFrame):
    """
    Plot standard deviation of daily temperatures for each month.
    """
    # Make sure Month column exists
    if 'Month' not in df.columns:
        df['Month'] = df['Date'].dt.month

    # Group by Month and compute standard deviation
    std_by_month = df.groupby('Month')['Temp'].std().reset_index()

    # Plot
    plt.figure(figsize=(10, 5))
    sns.barplot(data=std_by_month, x='Month', y='Temp', palette="Blues")

    plt.title("Standard Deviation of Daily Temperatures by Month over years")
    plt.xlabel("Month")
    plt.ylabel("Temperature Std (°C)")
    plt.xticks(ticks=range(0, 12), labels=[
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ])
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/std_dev_temp_vs_month.png")
    # plt.show()


def plot_monthly_avg_temp(df):
    grouped = df.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std']).reset_index()
    grouped.rename(columns={'mean': 'AvgTemp', 'std': 'TempStd'}, inplace=True)

    plt.figure(figsize=(12, 6))
    countries = grouped['Country'].unique()

    for country in countries:
        country_data = grouped[grouped['Country'] == country]
        plt.errorbar(
            country_data['Month'],
            country_data['AvgTemp'],
            yerr=country_data['TempStd'],
            label=country,
            capsize=4,
            marker='o',
            linestyle='-'
        )

    plt.title("Average Monthly Temperature by Country")
    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    plt.xticks(range(1, 13))
    plt.legend(title="Country")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/avg_temp_vs_country.png")
    plt.show()


# expects data from Israel only
def evaluate_polynomial_fitting_israel(israel_data: pd.DataFrame):
    # Get X and y
    israel_data = israel_data[["DayOfYear", "Temp"]].copy()

    sampled_df = israel_data.sample(frac=0.75, random_state=RANDOM_SEED)
    test_sampled_df = israel_data.drop(sampled_df.index)

    # train data
    train_x = sampled_df["DayOfYear"].values
    train_y = sampled_df["Temp"].values

    # test data
    test_x = test_sampled_df["DayOfYear"].values
    test_y = test_sampled_df["Temp"].values

    test_errors = []

    # Loop over k in 1 to 10
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(train_x, train_y)
        loss = model.loss(test_x, test_y)
        test_errors.append(round(loss, 2))
        print(f"k={k} | Test MSE: {round(loss, 2)}")

    # Plotting
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(1, 11), test_errors, color="skyblue")
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # x position (center of bar)
            height,  # y position (top of bar)
            f'{height:.2f}',  # text (formatted to 2 decimals)
            ha='center',  # horizontal alignment
            va='bottom'  # vertical alignment
        )
    plt.xlabel("Polynomial Degree (k)")
    plt.ylabel("Test MSE Loss")
    plt.title("Test Error of Polynomial Fitting on Israel Data")
    plt.xticks(list(range(1, 11)))
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/polynomial/loss_vs_poly_degree.png")
    plt.show()


def eval_israel_model_on_diff_countries(df: pd.DataFrame):
    # Fit a model by israel
    israel_data = df[df['Country'] == "Israel"]
    israel_data = israel_data[["DayOfYear", "Temp"]].copy()

    train_x = israel_data["DayOfYear"].values
    train_y = israel_data["Temp"].values

    model = PolynomialFitting(MINIMAL_K)
    model.fit(train_x, train_y)

    losses: Dict = {}

    # Test a model by other countries
    countries = [c for c in df['Country'].unique() if c != 'Israel']
    for country in countries:
        country_data = df[df['Country'] == country]
        test_x = country_data["DayOfYear"].values
        test_y = country_data["Temp"].values
        loss = model.loss(test_x, test_y)
        losses[country] = round(loss, 2)

    sorted_losses = dict(sorted(losses.items(), key=lambda item: item[1]))
        # Plotting the results
    plt.figure(figsize=(12, 6))
    colors = ['orange', 'green', 'red'] * (len(sorted_losses) // 3) + ['orange', 'green', 'red'][
                                                                      :len(sorted_losses) % 3]
    bars = plt.bar(sorted_losses.keys(), sorted_losses.values(), color=colors)
    # Add text above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # x position (center of bar)
            height,  # y position (top of bar)
            f'{height:.2f}',  # text (formatted to 2 decimals)
            ha='center',  # horizontal alignment
            va='bottom'  # vertical alignment
        )
    plt.title(f"Model Loss by Country (Trained on Israel, k={MINIMAL_K})")
    plt.xlabel("Country")
    plt.ylabel("Loss (MSE)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/polynomial/israel_model_loss_by_country.png")
    plt.show()



if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    israel_df = df[df['Country'] == "Israel"]
    plot_temp_by_day(israel_df)
    plot_monthly_temp_std(israel_df)
    # Question 4 - Exploring differences between countries
    plot_monthly_avg_temp(df)
    # Question 5 - Fitting model for different values of `k`
    evaluate_polynomial_fitting_israel(israel_df)
    # Question 6 - Evaluating fitted model on different countries
    eval_israel_model_on_diff_countries(df)
