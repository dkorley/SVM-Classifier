import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

class SVMClassifier:
    def __init__(self, C: int = 50, alpha: int = 1, tol: float = 1e-5, kmax: int = 1000) -> None:
        self.C = C
        self.alpha = alpha
        self.tol = tol
        self.kmax = kmax
        self.v = None
        self.b = None

    def proj_v(self, Y, w):
        """Projection onto the subspace orthogonal to Y."""
        n = w.shape[0]
        return -(Y.T @ w) * Y / n + w

    def proj_first_octant(self, w):
        """Projection onto the first octant clipped at C."""
        return np.clip(w, 0, self.C)

    def dykstra(self, Y, v):
        """Dykstra's algorithm for projections."""
        x = np.zeros_like(v)
        y = np.zeros_like(v)
        diff = 2 * self.tol
        q = v
        iter_count = 0
        
        while diff > self.tol and iter_count < self.kmax:
            qold = q
            p = self.proj_v(Y, q + x)
            x = q + x - p
            q = self.proj_first_octant(p + y)
            y = p + y - q
            diff = np.linalg.norm(qold - q)
            iter_count += 1

        return q 

    def fit(self, X, Y):
        """Train the SVM model."""
        m, n = X.shape
        Y = Y.reshape(-1, 1)
        A = Y * X
        Q = A @ A.T
        L = np.abs(np.linalg.eigvals(Q).max())
        t = self.alpha / L
        u = np.random.rand(m, 1)
        k = 1
        diff = 2 * self.tol
        c = np.ones((m, 1))

        while diff > self.tol and k < self.kmax:
            gradf = Q @ u - c
            u_old = u
            y = u - t * gradf
            u = self.dykstra(Y, y)
            k += 1
            diff = np.linalg.norm(u - u_old)

        self.v = A.T @ u
        z = (Y * u).flatten()
        
        # Find support vectors
        i_pos = np.where((z > 0) & (z < self.C))[0]
        j_neg = np.where((z < 0) & (z > -self.C))[0]

        if len(i_pos) > 0 and len(j_neg) > 0:
            i = i_pos[0]
            j = j_neg[0]
            self.b = (-(X[i] + X[j]) @ self.v + 1) / 2
        else:
            print("Warning: No valid support vectors found. Defaulting b to 0.")
            self.b = 0.0

    def predict(self, X):
        """Make predictions on new data."""
        if self.v is None or self.b is None:
            raise ValueError("Model not trained. Please call fit method first.")

        Z = np.dot(X, self.v) + self.b
        return np.where(Z >= 0, 1, -1)

    def score(self, X, y):
        """Compute the accuracy of the model."""
        predictions = self.predict(X)
        accuracy = 100 * np.mean(predictions == y)
        return accuracy

if __name__ == "__main__":
    # Generate some random data
    np.random.seed(0)
    X = np.random.uniform(0, 1, (100, 4))
    y = np.random.choice([-1, 1], 100)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

    C = [0.1, 1, 10, 100, 1000]
    for c in C:
        svm = SVMClassifier(C=c)
        svm.fit(X_train, y_train)
        accuracy = svm.score(X_test, y_test)
        print(f"C={c}, Accuracy: {accuracy}")

