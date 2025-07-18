from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface
import numpy as np



# rtde_r = RTDEReceiveInterface(hostname="134.28.40.74")

# # getting output state 

# x = rtde_r.getDigitalOutState(1)

# s = 'DigitalOutState'

# x = getattr(rtde_r, 'getDigitalOutState')(1)

# print(x)

# setting output state

rtde_io= RTDEIOInterface(hostname="134.28.40.74")


rtde_io.setStandardDigitalOut(1, True)

rtde_io.disconnect()





