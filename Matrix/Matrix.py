#import numpy.matlib
import numpy as np


def primitive_matrix_checks():
    str = "Primitive Matrix Operations"
    print(str + "\n" + len(str) * "-")
    print("A = \n{}\n"  .format(A))
    print("B = \n{}\n"  .format(B))
    print("C = \n{}\n"  .format(C))
    print("A + B = \n{}\n" .format(A+B))
    print("A * B = \n{}\n" .format(A*B))
    print("A / B = \n{}\n" .format(A/B))
    print("A / C = \n{}\n" .format(A/C))
    print("1 / A = \n{}\n" .format(1/A))
    print("C / A = \n{}\n" .format(C/A))
    print("B / A = \n{}\n" .format(B/A))


def basic_matrix_checks():
    str = "Basic Matrix Operations"
    print(str + "\n" + len(str) * "-")
    print("inv(AB) = \n{}\n" .format(np.linalg.inv(A * B)))
    print("inv(B).inv(A) = \n{}\n" .format(np.linalg.inv(B) * np.linalg.inv(A)))
    print("inv(ABC) = \n{}\n" .format(np.linalg.inv(A * B * C)))
    print("inv(C).inv(B).inv(A) = \n{}\n" .format(np.linalg.inv(C) * np.linalg.inv(B) * np.linalg.inv(A)))
    print("inv(trans(A)) = \n{}\n" .format(np.linalg.inv(np.transpose(A))))
    print("trans(inv(A)) = \n{}\n" .format(np.transpose(np.linalg.inv(A))))
    eigen_array, eigen_matrix = np.linalg.eig(A)
    print("Eigen Vector (A) = \n{}\n" .format(eigen_array))
    print("Sum of eigen values = \n{}\n" .format(eigen_array[0] + eigen_array[1]))
    print("trace(A) = \n{}\n" .format(np.trace(A)))


def orthogonal_check(a):
    a_at = a * a.transpose()
    i2 = np.identity(2)
    if np.array_equal(a_at, i2):
        print("Yes the matrix {} is orthogonal\n" .format(a))
    else:
        print("Matrix a = \n{} is NOT orthogonal"  .format(a))
        print("a * trans(a) = \n{}\n"  .format(a_at))
        print("i2 = \n{}\n" .format(i2))


if __name__ == '__main__':
    A = np.matrix('2, 7; -1, -6')
    B = np.matrix('2, 1; 3, 2')
    C = np.matrix('1, 2; 3, 4')

    primitive_matrix_checks()
    basic_matrix_checks()
    D = np.matrix('1, 0; 0, -1')
    orthogonal_check(D)
