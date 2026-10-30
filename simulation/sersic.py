import numpy as np
import matplotlib.pyplot as plt

step = 0.5 #make sure step in extended_source and here are same
point = 10

def create_source_grid():
    coordinates = []

    values = np.linspace(
        -point,
        point,
        int(2 * point / step) + 1
    )

    for x in values:
        for y in values:
            coordinates.append([x, y])

    return coordinates

#we are going align the whole source system 
# for ellipse because , we are going to align
# major axis with x axis

def rotate_source(phi):
    positions = create_source_grid()
    rotated_grid =[]
    for i in range(len(positions)):
        x_prime = positions[i][0] * np.cos(phi * (np.pi / 180)) + positions[i][1] * np.sin(phi * (np.pi/180))
        y_prime = -(positions[i][0]) * np.sin(phi * (np.pi / 180)) + positions[i][1] * np.cos(phi * (np.pi/180))
        rotated_grid.append([x_prime,y_prime])
    return rotated_grid

def ellipse_radius(phi, q):
    positions = create_source_grid()
    ellipse_radius_mapping = {}

    angle = phi * np.pi / 180

    for k in range(len(positions)):
        x = positions[k][0]
        y = positions[k][1]

        x_prime = x * np.cos(angle) + y * np.sin(angle)
        y_prime = -x * np.sin(angle) + y * np.cos(angle)

        r_ell_sq = x_prime**2 + (y_prime**2 / q**2)
        r_ell = np.sqrt(r_ell_sq)

        ellipse_radius_mapping[(x, y)] = r_ell

    return ellipse_radius_mapping
    
def sersic_brightness(phi , q,R_e , I_e,n ):
    #previously we used gaussian decay , but for modelling
    #approx real galaxy we use this decay method
    #that assigns brightness to each pixels
    #plausibly
    maps = ellipse_radius(phi,q)
    b_map ={}
    b_n = 2*n - 1/3 #controls , so that R_e stays 
    #effective radius no matter for any values of n
    for keys in maps:
        intermediate = -b_n * ((maps[keys]/R_e)**(1/n)-1)
        image = I_e * np.exp(intermediate)
        b_map[keys] = image
    return b_map

#Visulaising
phi = 45
q = 0.6
R_e = 3
I_e = 1
n = 1

b_map = sersic_brightness(phi, q, R_e, I_e, n)

positions = create_source_grid()


size = int(2 * point / step) + 1
image = np.zeros((size, size))

for i in range(len(positions)):
    x = positions[i][0]
    y = positions[i][1]

    row = i // size
    col = i % size

    image[row, col] = b_map[(x, y)]

plt.imshow(
    image,
    extent=[-point, point, -point, point],
    origin="lower"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Sérsic Source")
plt.show()  
    

 