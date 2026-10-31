import numpy as np
import matplotlib.pyplot as plt
from point_mass import lensing

def create_circle(beta, einstein_radius):
    #einstein radius is theta_e which is direction offset due to lens
    coordinates =[]
    step = 0.1
    for i in np.arange(einstein_radius , -einstein_radius-1 , -step):
        for j in np.arange(einstein_radius , -einstein_radius-1, -step):
            if np.abs((i**2 + j**2) - einstein_radius**2) < 0.5:
                # approximation of x^2 + y^2 = R^2
                coordinates.append([i,j])
    x=[]
    y=[]            
    for coordinate in coordinates:
        x.append(coordinate[0])
        y.append(coordinate[1])
    plt.scatter(x,y)
    plt.scatter(0,0)
    plt.text(0.1 ,0.1 , "Lens")
    theta_plus , theta_minus = lensing(beta , einstein_radius)
    plt.scatter(beta ,0)
    plt.text(beta +0.1 , 0.1 , "S")
    plt.scatter(theta_plus , 0)
    plt.text(theta_plus+0.1 , 0.1 ,"I+")
    plt.scatter(theta_minus ,0)
    plt.text(theta_minus + 0.1 , 0.1 , "I-")
    plt.show()   



'''
Till now what we did was point lens + point source
but evenntually the real world is not a point anyways
its mass is distributed over space and that is called
distributed mass so , this can affect the light
dramatically so each spacetime affects the light 
individually so next toy is obviously extended/
distributed source + point lens , consecutive to the
next will be the extended(source + lens)
'''
      