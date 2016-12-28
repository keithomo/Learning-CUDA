# import the python modules we need
import numpy as np
import pylab as plt
import struct


import os
import sys

path = "hermes"
fifo = open(path, "r")

tmp = fifo.read()
power = np.frombuffer(tmp,dtype = np.dtype('float32'))
print power
fifo.close()

