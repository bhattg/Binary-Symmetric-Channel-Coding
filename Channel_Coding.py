import math
import numpy as np
from numpy.random import binomial
from tqdm import tqdm

def rate(m, n):
    return log2(m)/n
def log_2(x):
    return np.log(x)/np.log(2)
def entropy(p):
    h  = (p*log_2(p)+ (1-p)*log_2(1-p))
    return -h
def capacity(p):
    return 1- entropy(p)

def generate_codebook(theta, n, r):
    M = math.ceil(pow(2, n*r))
    codebook=[]
    for i in tqdm(range(M)):
        codebook.append(binomial(n=1, p=theta, size=n))
    return codebook

def channel(p, x):
    # p represents the error transition probability 
    if x==0:
        return binomial(1, p, 1)
    else:
        return 1-binomial(1, p, 1)

def vector_channel(x_in, p):
    x_in = list(x_in)
    l=len(x_in)
    y_out = []
    for x in (x_in):
        y_out.append(channel(p, x)) 
    return y_out


def get_p_y(p_y_x, p_x):
    p_y_x = np.asarray(p_y_x)
    p_x = np.asarray(p_x)
    p_y = np.matmul(p_x, p_y_x)
    return p_y

def get_joint_pdf(p_y_x, p_x):
    #p_x -- (1, 2) list [a, b]
    p_x = (np.asarray([p_x, p_x])).T
    return np.multiply(p_x, p_y_x)

def get_H_y(p_y):
    # y_x is a (2, 2)
    # p_x is (1, 2)
    return entropy(p_y[0])

def get_H_x(p_x):
    return entropy(p_x[0])

def get_H_x_y(p_joint):
    return -np.sum(np.multiply(p_joint, log_2(p_joint)))    

def compute_sample_mean(p_, x_in):
    s = 0
    for x in x_in:
        s+=log_2(p_[x])
    return -s/len(x_in)

def compute_joint_sample_mean(p_joint, x, y):
    s=0
    for i in range(len(x)):
        s+=log_2(p_joint[x[i]][y[i]])    
    return -s/len(x)

def check_both_typicality(p_joint, x, y, Hxy):
    sample_mean = compute_joint_sample_mean(p_joint, x, y)
    if abs(sample_mean-Hxy)<epsilon:
        return True
    else:
        return False

def check_x_typical(p_x, x_in, Hx):
    # given input sequence x_in 
    # it should be a list
    sample_mean = compute_sample_mean(p_x, x_in)
    if abs(sample_mean-Hx)<epsilon:
        return True
    else:
        return False

def check_y_typicality(p_y, y_out, Hy):
    #p(y1, y2, ....) = 2^nH(y)
    sample_mean=compute_sample_mean(p_y, y_out)
    if abs(sample_mean-Hy)<epsilon:
        return True
    else:
        return False
    
def check_joint_typicality(x_in, y_in, H_x, H_y, H_x_y, p_x, p_y, p_joint):
    return check_x_typical(p_x, x_in, H_x) and check_y_typicality(p_y, y_in, H_y) and check_both_typicality(p_joint, x_in, y_in, H_x_y)
   
def decoder(codebook, y_in, H_x, H_y, H_x_y, p_x, p_y, p_joint):
    possible_senders= []
    for i in tqdm(range(len(codebook))):
        if check_joint_typicality(codebook[i], y_in, H_x, H_y, H_x_y, p_x, p_y, p_joint) :
            possible_senders.append(i)
            if len(possible_senders)>=2:
                break
    return possible_senders

