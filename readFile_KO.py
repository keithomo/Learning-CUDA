# import the python modules we need
import numpy as np
import pylab as plt
import struct

# I know a priori that I want one row to be this long
rowLen = 1024

# open the file, and read raw data, as bytes, into a temporary array 
f = open('known_real.dat')
tmp = f.read()
f.close()

# make array
power = np.frombuffer(tmp,dtype = np.dtype('float32'))
# convert into a numpy 2d array, with the number of rows calculated automatically
power_a = np.transpose(np.reshape(power, (-1,rowLen)))

# plot as log array
plt.xlabel('frequency')
plt.ylabel('time')
plt.imshow(np.log10(power_a),origin='lower', cmap='Greys_r')
plt.colorbar()
plt.show()

# plot as an array
plt.xlabel('frequency')
plt.ylabel('time')
plt.imshow(power_a,origin='lower', cmap='Greys_r')
plt.colorbar()
plt.show()



import numpy as np
import pylab as plt
import struct
rowLen = 1024
f = open('spectro5.dat')
tmp = f.read()
f.close()
power = np.frombuffer(tmp,dtype = np.dtype('float32'))
power_a = np.transpose(np.reshape(power, (-1,rowLen)))
plt.ylabel('frequency')
plt.xlabel('time')
plt.imshow(np.log10(power_a),origin='lower', cmap='Greys_r')
plt.colorbar()
plt.show()


#################################################################
#import numpy as np
#import pylab as plt
#import struct
#
## I know a priori that I want one row to be this long
#rowLen = 1024
#
## open the file, and read raw data, as bytes, into a temporary array 
#f = open('known_real.dat')
#tmp = f.read()
#f.close()
#
## make array
#power = np.frombuffer(tmp,dtype = np.dtype('float32'))
#x = np.array([1])
#for i in range(1024):
#	x = np.append(x,[i])
#
#x = np.delete(x, 0)
#plt.plot(x,power)
#plt.show()
#
