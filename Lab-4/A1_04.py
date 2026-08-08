def calculate_minkowski_distance(v1, v2, p):
    distance = 0

    for i in range(len(v1)):
        distance += abs(v1[i] - v2[i]) ** p

    distance = distance ** (1 / p)

    return distance


vector1 = [100, 200, 300]
vector2 = [110, 190, 310]

p = int(input("Enter p value: "))

result = calculate_minkowski_distance(vector1, vector2, p)

print("Distance =", result)