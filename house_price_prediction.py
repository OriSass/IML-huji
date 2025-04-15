from typing import NoReturn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick
from linear_regression import LinearRegression

RANDOM_SEED = 42
YEARS_BACK_TO_CONSIDER = 35
PLOT_DIR = "./plots"
COLUMNS_TO_DROP = ['sqft_living15', 'sqft_lot15', 'id', 'sqft_above', 'sqft_basement', 'date', 'lat', 'long', 'waterfront', 'view']


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """

    # Convert the 'date' column to datetime objects with proper error handling
    X['date'] = pd.to_datetime(X['date'], format='%Y%m%dT%H%M%S', errors='coerce')

    # Check for any NaT values (invalid dates)
    X = X.dropna(subset=['date'])  # Remove rows with invalid dates
    y = y[X.index]

    y = pd.to_numeric(y, errors='coerce')
    valid_y_mask = y.notna()
    y = y[valid_y_mask]
    X = X[valid_y_mask]

    # Get today's date
    # today = pd.Timestamp.now()

    # Calculate the year x years ago
    # x_years_ago_year = today.replace(year=today.year - YEARS_BACK_TO_CONSIDER).year

    # Ensure specified fields are numeric and drop rows with NaNs in them
    numeric_fields = [
        'sqft_living', 'sqft_lot',
        'floors', 'condition', 'grade',
        'yr_built', 'yr_renovated'
    ]
    for field in numeric_fields:
        X[field] = pd.to_numeric(X[field], errors='coerce')

    # X = X.dropna(subset=numeric_fields)
    X = X.dropna()
    y = y[X.index]

    X = X.drop(columns=COLUMNS_TO_DROP)
    y = y[X.index]

    filtered_x = X[
        (((X['yr_renovated'] >= X['yr_built']) & (X['yr_renovated'] != 0)) | X['yr_renovated'] == 0) &
        (X['sqft_lot'] >= X['sqft_living'])
        ]
    filtered_y = y[filtered_x.index]
    return filtered_x, filtered_y


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    # Convert numeric columns to proper types
    numeric_fields = [
        'sqft_living', 'sqft_lot',
        'floors', 'condition', 'grade',
        'yr_built', 'yr_renovated'
    ]
    for field in numeric_fields:
        X[field] = pd.to_numeric(X[field], errors='coerce')

    # Fill NaNs (e.g., with median values)
    X[numeric_fields] = X[numeric_fields].fillna(X[numeric_fields].median())

    # Drop unused columns
    return X.drop(columns=COLUMNS_TO_DROP)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # Iterate through each feature (column) in the DataFrame X
    for feature in X.columns:
        # print(feature)
        x_feature = X[feature]

        # Skip non-numeric features
        if not np.issubdtype(x_feature.dtype, np.number):
            continue

        # Calculate Pearson correlation manually
        cov = np.cov(x_feature, y)[0, 1]
        std_x = np.std(x_feature, ddof=0)
        std_y = np.std(y, ddof=0)
        if std_x == 0 or std_y == 0:
            continue
        corr = cov / (std_x * std_y)

        # Align x_feature and y by index
        valid_indices = x_feature.index.intersection(y.index)
        x_vals = x_feature.loc[valid_indices]
        y_vals = y.loc[valid_indices]

        # Create scatter plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=x_vals, y=y_vals)

        # Title with feature name and Pearson correlation
        plt.title(f'{feature} vs Price\nPearson Correlation: {corr:.2f}')
        plt.xlabel(feature)
        plt.ylabel("Price")

        y_axis_format = mtick.FuncFormatter(lambda x, _: f'{int(x / 1000)}K')
        plt.gca().yaxis.set_major_formatter(y_axis_format)

        # Save the plot to the specified output path
        plt.savefig(f"{output_path}/{feature}_scatter_plot.png")

        # Close the plot to avoid overlap with the next plot
        plt.close()


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")  # dataframe - tabular data
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    # Number of rows
    n = len(X)
    # Set the proportion for training set
    train_size = 0.75  # 75% training, 25% testing
    # Shuffle the data and take a sample for the training set
    train_df = X.sample(frac=train_size, random_state=RANDOM_SEED)
    # Use the rest for the test set
    test_df = X.drop(train_df.index)

    # Extract target values for train and test
    train_y = y[train_df.index]  # Target values corresponding to the training set
    test_y = y[test_df.index]  # Target values corresponding to the test set

    # Find the index where the NaN value in test_y is located
    nan_index = test_y[test_y.isna()].index

    # Remove the NaN value from test_y and the corresponding row in test_df
    test_y = test_y.drop(nan_index)
    test_df = test_df.drop(nan_index)

    # After removing the row, check the result
    # print(f"NaN values in test_y after removal: {test_y.isna().sum()}")
    # print(f"Test data shape after removal: {test_df.shape}")

    # Question 3 - preprocessing of housing prices train dataset
    pre_processed_train_df, pre_processed_train_y = preprocess_train(train_df, train_y)
    rows_cleaned = train_df.shape[0] - pre_processed_train_df.shape[0]
    print(f"Pre-process cleaned: {rows_cleaned} rows")

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(pre_processed_train_df, pre_processed_train_y, PLOT_DIR)
    # Question 5 - preprocess the test data
    pre_processed_test_df = preprocess_test(test_df)

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:

    # create a linear regression model
    model = LinearRegression(include_intercept=True)

    # for results storage
    avg_losses = []
    std_losses = []
    percentages = list(range(10,101))
    # print(f"Train data shape: {pre_processed_train_df.shape}")
    # print(f"Train target shape: {pre_processed_train_y.shape}")
    # print(f"Test data shape: {pre_processed_test_df.shape}")
    # print(f"Test target shape: {test_y.shape}")
    for p in percentages:
        losses = []
        for _ in range(10):
            #   1) Sample p% of the overall training data
            fraction = p / 100
            sampled_train_df = pre_processed_train_df.sample(frac=fraction, random_state=RANDOM_SEED)

            #   2) Fit linear model (including intercept) over sampled set

            model.fit(sampled_train_df.to_numpy(), pre_processed_train_y[sampled_train_df.index].to_numpy())

            #   3) Test fitted model over test set
            #   4) Store average and variance of loss over test set
            loss = model.loss(pre_processed_test_df.to_numpy(), test_y.to_numpy())
            losses.append(loss)

        # Calculate average and standard deviation of the losses for this percentage
        avg_loss = np.mean(losses) # ממוצע
        std_loss = np.std(losses) # סטיית תקן
        avg_losses.append(avg_loss)
        std_losses.append(std_loss)
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # Plot the results

    lower = np.array(avg_losses) - 2 * np.array(std_losses)
    upper = np.array(avg_losses) + 2 * np.array(std_losses)

    print("Lower bound:", lower[:5])
    print("Upper bound:", upper[:5])
    print("Difference:", (upper - lower)[:5])

    plt.figure(figsize=(10, 6))
    plt.plot(percentages, avg_losses, label='Average Loss', color='blue')
    plt.fill_between(percentages,
                     lower,
                     upper,
                     color='blue', alpha=0.5, label="± 2 Std Dev (normalized)")
    plt.title("Normalized Loss vs. Training Data Size")
    plt.xlabel("Percentage of Training Data")
    plt.ylabel("Average Mean Squared Error (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{PLOT_DIR}/training_data_performance.png")
    plt.show()




