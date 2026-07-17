import pandas as pd

def get_binary_data(data):
    first_two = data.iloc[:2]

    binary_columns = []

    for col in data.columns:
        values = data[col].dropna().unique()
        if len(values) == 2:
            binary_columns.append(col)

    binary_data = first_two[binary_columns].copy()

    for col in binary_columns:
        values = data[col].dropna().unique()

        if not set(values).issubset({0, 1}):
            binary_data[col] = binary_data[col].replace({
                values[0]: 0,
                values[1]: 1
            })

    return binary_data


def calculate_similarity(v1, v2):
    f00 = f01 = f10 = f11 = 0

    for i in range(len(v1)):
        if v1[i] == 0 and v2[i] == 0:
            f00 += 1
        elif v1[i] == 0 and v2[i] == 1:
            f01 += 1
        elif v1[i] == 1 and v2[i] == 0:
            f10 += 1
        else:
            f11 += 1

    jc = f11 / (f01 + f10 + f11)
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)

    return f00, f01, f10, f11, jc, smc


data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="thyroid0387_UCI")

binary_data = get_binary_data(data)

v1 = binary_data.iloc[0].values
v2 = binary_data.iloc[1].values

f00, f01, f10, f11, jc, smc = calculate_similarity(v1, v2)

print("vector 1:", v1)
print("vector 2:", v2)

print("\nf00 =", f00)
print("f01 =", f01)
print("f10 =", f10)
print("f11 =", f11)

print("\njaccard coefficient:", jc)
print("simple matching coefficient:", smc)
