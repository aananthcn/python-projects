# Singular Value Decomposition Example

import numpy as np

# example taken from Video Tutorials - All in One
# https://www.youtube.com/watch?v=P5mlg91as1c
a = np.array([[2, 4,],
              [1, 3,],
              [0, 0,],
              [0, 0,]])

# set numpy printing options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

# Full SVD is taught more often. Here is a good explination of the different
# http://www.cs.cornell.edu/Courses/cs322/2008sp/stuff/TrefethenBau_Lec4_SVD.pdf
print "--- FULL ---"
U, s, VT = np.linalg.svd(a, full_matrices=True)

# Added by Aananth
m_U = np.matrix(U)
m_s = np.matrix([[s[0],    0],
                 [0,    s[1]],
                 [0,       0],
                 [0,       0]
                 ])
m_VT = np.matrix(VT)

print "U:\n {}".format(U)
print "s:\n {}".format(s)
print "VT:\n {}".format(VT)

# the reduced or trucated SVD operation can save time by ignoring all the
# extremly small or exactly zero values. A good blog post explaing the benefits
# can be found here:
# http://blog.explainmydata.com/2016/01/how-much-faster-is-truncated-svd.html
print "--- REDUCED ---"

U, s, VT = np.linalg.svd(a, full_matrices=False)

print "U:\n {}".format(U)
print "s:\n {}".format(s)
print "VT:\n {}".format(VT)

# Reverse validation by Aananth
A = m_U*m_s*m_VT
print("Orginal Mat = \n{}" .format(a))
print("Reverse SVD = \n{}" .format(A))