import numpy as np
from numpy import linalg as LA
import os,sys
import argparse
"""
parser = argparse.ArgumentParser(description='clustering agreement measures')

parser.add_argument('--data',  help='data file directory')
parser.add_argument('--data2',  help='data file directory')
parser.add_argument('--data3',  help='data file directory')
parser.add_argument('--data4',  help='data file directory')

args = parser.parse_args()

Y = np.loadtxt(open(args.data, "rb"), delimiter=" ")
Y_true = np.loadtxt(open(args.data2, "rb"), delimiter=" ")
X = np.loadtxt(open(args.data4, "rb"), delimiter=" ")
X_true = np.loadtxt(open(args.data4, "rb"), delimiter=" ")

Ym,Yn = Y.shape
Xm,Xn = X.shape

"""


def Frobenius_norm(A):
    return (np.trace(np.transpose(A).dot(A)))**0.5


def Frobenius_norm2(A):
    return LA.norm(A)


def F1(A,A2):
    temp = 0
    for i in range(A.shape[0]):
        temp+= (2 * np.transpose(A[i,:]).dot(A2[i,:]))/((A[i,:].dot(np.transpose(A[i,:]))+A2[i,:].dot(np.transpose(A2[i,:]))))
    
    return temp/A.shape[0]


def I_cos(A,A2):
    return Frobenius_norm(np.transpose(A).dot(A2))**2/(Frobenius_norm(A.dot(np.transpose(A)))*Frobenius_norm(A2.dot(np.transpose(A2))))


def I_cos2(A,A2):
    return Frobenius_norm2(np.transpose(A).dot(A2))**2/(Frobenius_norm2(A.dot(np.transpose(A)))*Frobenius_norm2(A2.dot(np.transpose(A2))))


def I_sub(A,A2):
    return Frobenius_norm(np.transpose(A).dot(A2))/(Frobenius_norm(A)*Frobenius_norm(A2))


def I_sub2(A,A2):
    return Frobenius_norm2(np.transpose(A).dot(A2))/(Frobenius_norm2(A)*Frobenius_norm2(A2))



"""
average of the measures:
"""
def avg_F1(Y,Y_true,X,X_true):
    return 0.5*(F1(Y,Y_true)+F1(X,X_true))


def avg_Icos(Y,Y_true,X,X_true):
    return 0.5*(I_cos(Y,Y_true)+I_cos(X,X_true))


def avg_Isub(Y,Y_true,X,X_true):
    return 0.5*(I_sub(Y,Y_true)+I_sub(X,X_true))


np.random.seed(1)
A = np.random.randint(2, size=(20,30))
np.random.seed(31)
A2 = np.random.randint(2, size=(20,30))


np.random.seed(2)
B = np.random.randint(2, size=(20,30))
np.random.seed(32)
B2 = np.random.randint(2, size=(20,30))

#Am,An = A.shape
#A2m,A2n = A2.shape


print(A)
print(A2)
print(B)
print(B2)
print("F1 :",F1(A,A2))
print("I_sub  :",I_sub(A,A2))
print("I_sub2 :",I_sub2(A,A2))
print("I_cos  :",I_cos(A,A2))
print("I_cos2 :",I_cos2(A,A2))
print("avg_F1 :",avg_F1(A,A2,B,B2))
print("avg_Icos :",avg_Icos(A,A2,B,B2))
print("avg_Isub :",avg_Isub(A,A2,B,B2))

