import random
import numpy
from math import *

def random_walk():
    alpha = random.uniform(-pi,pi)
    beta = random.uniform(-pi,pi)
    gama = random.uniform(-pi,pi)
    Rx = numpy.matrix([[1.,0.,0.],[0.,cos(alpha),-sin(alpha)],[0.,sin(alpha),cos(alpha)]])
    Ry = numpy.matrix([[cos(beta),0.,sin(beta)],[0.,1.,0.],[-sin(beta),0.,cos(beta)]])
    Rz = numpy.matrix([[cos(gama),-sin(gama),0.],[sin(gama),cos(gama),0.],[0.,0.,1.]])
    return Rz*Ry*Rx