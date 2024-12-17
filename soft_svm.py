import numpy as np
import matplotlib.pyplot as plt

def proj_v(Y, w):
    n = w.shape[0]
    p = -(Y.T @ w) * Y/n + w
    return p

def proj_first_octant(w, C = 50.):
    v = w.copy()
    n = v.shape[0]
    for i in range(0,n):
        if v[i] < 0:
            v[i] = 0
        elif v[i] > C:
            v[i] = C
    return v

def dykstra(Y, v, tol=1e-5, max_iter=1000,C = 50):
    # Initialize the previous and current iterate
    x = np.zeros_like(v)
    y = np.zeros_like(v)

    
    # Initialize the error difference and iteration counter
    diff = 2 * tol
    iter_count = 0
    
    q = v
    # Perform the Dykstra projection
    while diff > tol and iter_count < max_iter:
        # Perform the projection onto Y
        qold = q
        p = proj_v(Y, q+x)
        x = q+x-p
        q = proj_first_octant(p+y,C)
        y = p+y-q
        # Update the error difference and iteration counter
        diff = np.linalg.norm(qold - q)
        iter_count += 1
    
    return q 

def svm1(X, Y, C=50, alpha=1, tol=1e-5, kmax=1000):
    m, n = X.shape
    A = Y * X
    Q = A @ A.T
    L = np.abs(np.linalg.eigvals(Q).max())  # maximum eigenvalue of Q
    t = alpha / L
    u = np.random.rand(m, 1)
    k = 1  # the counter
    diff = 2 * tol  # initialize the error difference
    c = np.ones((m, 1))
    
    # Main loop to solve for u using Dykstra projection
    while diff > tol and k < kmax:
        gradf = Q @ u - c
        u_old = u
        y = u - t * gradf
        u = dykstra(Y, y, C)
        k += 1
        diff = np.linalg.norm(u - u_old)
    
    # Compute the optimal weights
    v = A.T @ u
    gamma = 1 / (np.linalg.norm(v))
    z = (Y * u).flatten()
    
    # Find indices for support vectors
    support_indices_pos = np.where((z > 0) & (z < C))[0]
    support_indices_neg = np.where((z < 0) & (z > -C))[0]
    
    if len(support_indices_pos) > 0 and len(support_indices_neg) > 0:
        # Select one positive and one negative support vector to compute b
        i = support_indices_pos[0]
        j = support_indices_neg[0]
        
        b = (-(X[i] + X[j]) @ v + 1) / 2
    else:
        # If no valid support vectors, set b to zero (default behavior)
        print("Warning: No valid support vectors found. Defaulting b to 0.")
        b = 0.0
    
    return v, b, gamma
   
    
def main():
    X = np.array([[1.75, 2.8],
               [0.25, 2.1],
               [1, 1.5],
               [0.6, 0.6],
               [1.9, 1.1],
               [1, 1.75],
               [1.9, 1.4],
               [2.5, 1.5]])

    #normalize X array
    X = X / np.max(X, axis=0)
     
    Y = np.array([[-1., -1, -1, 1, 1, -1, 1, 1]]).T
    C = 50.
    v, b, g = svm1(X, Y,C)
    print("Vector V: \n", v)
    print("b: \n", b)
    print("gamma: \n", g)
    
    # x = np.linspace(0, 1, 100)
    # y = (-b - v[0]*x) / v[1]
    # y1 = (1 - b.flatten() - v[0]*x) / v[1]
    # y2 = (-1 - b.flatten() - v[0]*x) / v[1]
    
    # fig, ax = plt.subplots()
    # ax.plot(x, y, label='separating hyperplane')
    # ax.plot(x, y1)
    # ax.plot(x, y2)
    # ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, label='feature variable')
    # ax.legend()
    # plt.show()
if __name__ == '__main__':
    main()