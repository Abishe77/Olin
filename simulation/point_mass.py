import numpy as np
import matplotlib.pyplot as plt

def lensing(beta ,theta_e):
    #beta is where the source would appear without lensing
    #theta is observed angular position/direction of the image on the sky
    #(lens is not a glass or tlescope) its any celestial body or anything
    #in between us and the source
    theta_plus = float((beta + np.sqrt(beta **2 + (4 * theta_e**2)))/2)
    theta_minus = float((beta - np.sqrt(beta **2 + (4 * theta_e**2)))/2)
    return theta_plus , theta_minus
'''
final = lensing(0.5 ,1)
print(final)
count =1
for keys in final:
    plt.scatter(0,count)
    plt.text(0,count+0.1 ,"L "+ str(count))

    plt.scatter(keys , count)
    plt.text(keys , count+0.1 , "S "+str(count))

    plt.scatter(final[keys][0] ,count)
    plt.text(final[keys][0],count+0.1 ,"I+ "+str(count ))

    plt.scatter(final[keys][1] ,count)
    plt.text(final[keys][1],count+0.1 , "I- "+str(count))
    count+=1

    

plt.show()    



key = 2

Images = final[key]
plt.scatter(0,0)
plt.text(0,0.01 ,"Lens")

plt.scatter(key , 0)
plt.text(key,0.01 ,"Source")

plt.scatter(Images[0] , 0)
plt.text(Images[0] , 0.01 , "Image +")

plt.scatter(Images[1] , 0)
plt.text(Images[1] , 0.01 , "Image -")

plt.show()'''





