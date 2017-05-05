import numpy as np
import matplotlib.pyplot as plt

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def make_matrix(x, y):
    return np.array([
        [x, y],
        [1, 0]
    ])

def magnitude(mat):
    return sum(abs(v) for v in mat.flat)

ones = np.array([1, 1])
def get_sign(mat):
    """pos -> 1, neg -> 0"""
    return 1 if mat.dot(ones)[0] > 0 else 0

def test_color(mat):
    MAGNITUDE_THRESHOLD = 1e5
    power = np.copy(mat)
    sum_signs = 0
    for i in range(50):
        sum_signs += get_sign(power)
        power = power.dot(mat)
    print sum_signs
    if sum_signs < 10:
        return 'b'
    elif sum_signs < 30:
        return 'm'
    elif sum_signs <= 50:
        return 'y'
    else:
        print 'ERROR'

def get_signs(mat):
    power = np.copy(mat)
    sum_signs = 0
    for i in range(50):
        sum_signs += get_sign(power)
        power = power.dot(mat)
    return sum_signs


    # power = np.linalg.matrix_power(mat, 50)
    # if magnitude(power) > MAGNITUDE_THRESHOLD:
    #     return 'b'
    # return 'c'

if __name__ == '__main__':
    xs = []
    ys = []
    cs = []
    for x in frange(-3, 3, 0.02):
        for y in frange(-2, 2, 0.02):
            mat = make_matrix(x, y)
            # color = test_color(mat)
            signs = get_signs(mat)
            xs.append(x)
            ys.append(y)
            cs.append(signs)
    # plt.scatter(xs, ys, c=cs)
    plt.scatter(xs, ys, c=cs)
    plt.show()