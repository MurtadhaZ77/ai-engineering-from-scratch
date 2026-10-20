import math
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())
    def angle_between(self, other):
        cos_theta = self.cosine_similarity(other)
        radians = math.acos(cos_theta)
        degrees = math.degrees(radians)
        return degrees

    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])
print(f"a + b = {a + b}")
print(f"a . b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")
print(f"angle between a and b = {a.angle_between(b):.4f} degrees")

class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n {rows_str}"

    def matmul(self, other):
        if self.cols != other.rows:
            raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])


    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        raise ValueError("determinant only handles 1x1 and 2x2 for now")

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

A = Matrix([[1, 2], [3, 4]])
print(A.determinant())
print(A.inverse_2x2())
print(A.matmul(A.inverse_2x2()))
S = Matrix([[1, 2], [2, 4]])
print(S.determinant())
# print(S.inverse_2x2())
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])


def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])


pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)
print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(weights)
print(pre_activation)
print(output)


def random_matrix(rows, cols):
    return Matrix([
        [random.uniform(-1, 1) for _ in range(cols)]
        for _ in range(rows)
    ])


def dense_layer(W, x, b):
    pre_activation = W.matmul(x) + b
    return relu_matrix(pre_activation)


x = Matrix([[0.5], [0.8], [0.2]])
b1 = Matrix([[0.1], [0.1], [0.1], [0.1]])
b2 = Matrix([[0.1], [0.1]])

hidden = dense_layer(random_matrix(4, 3), x, b1)
output = dense_layer(random_matrix(2, 4), hidden, b2)
print("hidden shape:", hidden.shape)
print("output shape:", output.shape)
print(output)
