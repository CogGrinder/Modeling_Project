import numpy as np
import matplotlib.pyplot as plt 

def frobenius_dist(A, B):
    return np.sqrt(np.trace(np.transpose(A-B) * (A-B)))

def euclidean_dist(A, B):
    dist = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            dist += (A[i][j] - B[i][j]) ** 2
    return np.sqrt(dist)

def figure1():

    A = np.random.rand(3, 3)
    print(A)

    frobenius_results = []
    euclidean_results = []

    n = 500
    for i in range(0, n):
        B = np.random.rand(3, 3)
        d1 = frobenius_dist(A, B)
        d2 = euclidean_dist(A, B)
        frobenius_results.append(d1)
        euclidean_results.append(d2)
        # print("d1:", d1, "/ d2:", d2)

    calc_indices = np.array(list(range(0, n)))
    frobenius_results = np.array(frobenius_results)
    euclidean_results = np.array(euclidean_results)

    plt.scatter(calc_indices, frobenius_results, color="blue", label="Frobenius", marker="*", alpha=0.4)
    plt.scatter(calc_indices, euclidean_results, color="red", label="Euclidean", marker="*", alpha=0.4)
    plt.hlines(np.mean(frobenius_results), color="blue", xmin=0, xmax=n, label="Frobenius dist mean")
    plt.hlines(np.mean(euclidean_results), color="red", xmin=0, xmax=n, label="Euclidean dist mean")
    plt.title("Comparison of Frobenius and Euclidean distance over " + str(n) + " randomized computations")
    plt.xlabel("Computations")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(axis="x")
    plt.show()

def figure2():

    A = np.zeros((3, 3))
    print(A)

    frobenius_results = []
    euclidean_results = []

    sigma = 0.01
    n = 1000
    for i in range(0, n):
        B = A + np.random.randn(3, 3) * sigma
        d1 = frobenius_dist(A, B)
        d2 = euclidean_dist(A, B)
        frobenius_results.append(d1)
        euclidean_results.append(d2)

    calc_indices = np.array(list(range(0, n)))
    frobenius_results = np.array(frobenius_results)
    euclidean_results = np.array(euclidean_results)

    plt.scatter(calc_indices, frobenius_results, color="blue", label="Frobenius", marker="*", alpha=0.4)
    plt.scatter(calc_indices, euclidean_results, color="red", label="Euclidean", marker="*", alpha=0.4)
    plt.hlines(np.mean(frobenius_results), color="blue", xmin=0, xmax=n, label="Frobenius dist mean")
    plt.hlines(np.mean(euclidean_results), color="red", xmin=0, xmax=n, label="Euclidean dist mean")
    plt.title("Comparison over " + str(n) + " randomized iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(axis="x")
    plt.show()

if __name__=="__main__":
    # figure1()
    figure2()