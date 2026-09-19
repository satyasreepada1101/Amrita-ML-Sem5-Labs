import pandas as pd

def perform_binning(data, column_name, bin_type="equal_width", no_of_bins=5):

    # equal width binning
    if bin_type == "equal_width":
        binned_column = pd.cut(
            data[column_name],
            bins=no_of_bins,
            labels=False,
            include_lowest=True
        )

    # equal frequency binning
    elif bin_type == "equal_frequency":
        binned_column = pd.qcut(
            data[column_name],
            q=no_of_bins,
            labels=False,
            duplicates="drop"
        )

    else:
        print("invalid binning type")
        return None
    return binned_column


df = pd.read_csv("features.csv")

column_name = "top_left_R_mean"

equal_width = perform_binning(
    df,
    column_name,
    "equal_width",
    5
)

equal_frequency = perform_binning(
    df,
    column_name,
    "equal_frequency",
    5
)

print("\nEqual Width Binning")
print(equal_width.head())

print("\nEqual Frequency Binning")
print(equal_frequency.head())