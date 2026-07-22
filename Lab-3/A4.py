# minkowski is given as (sum(|P1(i)-P2(i)|)^p)^(1/p)
# p1 and p2 r vectors, and p is the order

def minkowski_dist(p1, p2, p):
    distance = 0

    for i in range(len(p1)):
        distance += abs(p1[i] - p2[i]) ** p

    return distance ** (1 / p)

p1 = [1, 2, 3]
p2 = [4, 5, 6]

distance1 = minkowski_dist(p1, p2, 1)
distance2 = minkowski_dist(p1, p2, 2)

print("minkowski dist (p=1) - manhattan:", distance1)
print("minkowski dist (p=2) - euclidean:", distance2)
