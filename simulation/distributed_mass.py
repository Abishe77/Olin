import numpy as np
import matplotlib.pyplot as plt
from extended_source import interpolate , see_galaxy #same func so we can resue

step = 0.5 #make sure step in extended_source and here are same
point = 10

def create_lens_grid():
    coordinates =[]
    for x in np.arange(-point , point+1 , step):
        for y in np.arange(-point , point+1 , step):
            coordinates.append([x,y])

    return coordinates
'''
prevously we did coordinates : distance from centre
: then each pixel brightness according to dist value
far away - low brightness  near to centre - high brightess

here the same mechanism , but the subtle diff is the
literal meaning , there it represented brightness
but here its "mass density" across the grid
so visually both looks same , but the literal meaning
is different
'''

def calculate_distance():
    lens_grid = create_lens_grid()
    #lens_grid is 2D list now
    result = {} # to store mappings
    for i in range(len(lens_grid)):
        r = np.sqrt((lens_grid[i][0] - 0)**2 + (lens_grid[i][1]-0)**2)
        result[tuple(lens_grid[i])] =r

    return result    

def surface_mass_density( A = 1 , std = 2):
    map = calculate_distance()
    mass_map ={}
    for keys in map:
        density = A * np.exp(-map[keys]**2 / (2 * std**2))
        mass_map[keys] = density

    return mass_map    

def see_lens():
    mass_map = surface_mass_density()
    x_values = []
    y_values = []
    for x , y in mass_map:
        x_values.append(x)
        y_values.append(y)
    #For numpy indexing
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    #to set dimensions
    width = int(((x_max - x_min) / step) +1)
    height = int(((y_max - y_min) / step) +1)
    lens = np.zeros((width,height))

    for x , y in mass_map:
        x_index = int((x-x_min)/ step)
        y_index = int((y-y_min)/ step)
        lens[x_index][y_index] = mass_map[(x,y)]

    return lens , x_min , y_min    

'''
Before we had a single point mass M at (0,0)
but here now its distributed across space
let that be (x' , y') now each x' , y' will
affect the path that light will propagate , 
intuitively it is ,at given light ray poisiton , the 
total lensing deflection is the combined effect
of the spacetime curvature produced by all the mass
in the lens , evaluated relative to that ray
'''

def calculate_deflection():
    mass_map = surface_mass_density()#the mass map
    grid = create_lens_grid() # the exact points
    deflection ={} #to store deflection for each point in mass grid
    cell_area = step ** 2 #for del_A
    for position in grid:
        alpha_x =0
        alpha_y =0
        for patches in mass_map:
            mass_patch = mass_map[patches] #has mass density
            del_x = position[0] - patches[0]
            del_y = position[1] - patches[1]
            r_sq = del_x**2 + del_y**2
            if r_sq ==0: #when position == patch
                continue
            x_contribution = (mass_patch * del_x) / r_sq * cell_area / np.pi
            y_contribution = (mass_patch * del_y) / r_sq * cell_area / np.pi 
            alpha_x += x_contribution
            alpha_y += y_contribution
        deflection[tuple(position)] =(alpha_x,alpha_y)

    return deflection

#mapping to the source plane to estimate where the light came from
deflections = calculate_deflection()
def map_to_source(deflections):
    store_positions ={} # (x,y) = (beta_x , beta_y)
    for x,y in deflections:
        alpha_x , alpha_y = deflections[(x,y)]
        beta_x = x - alpha_x
        beta_y = y - alpha_y
        store_positions[(x,y)] = (beta_x,beta_y)
    return store_positions

source_positions = map_to_source(deflections)

#Creating the lensed image
galaxy , x_min , y_min = see_galaxy()
def create_lensed_image(source_positions , galaxy , x_min , y_min):
    lensed_galaxy = np.zeros_like(galaxy) #to store actual image pixels
    for x , y in source_positions:
        beta_x ,beta_y = source_positions[(x,y)]
        index_x = (beta_x - x_min) / step
        index_y = (beta_y - y_min) / step
        if(
            index_x < 0 
            or index_x >= galaxy.shape[0] -1
            or index_y < 0
            or index_y >= galaxy.shape[1] -1
        ):
            brightness =0

        else:
            brightness = interpolate(galaxy , index_x,index_y)

        #to set acc to numpy dimensions
        x_output = int((x - x_min) / step)
        y_output = int((y - y_min) / step)
        lensed_galaxy [x_output , y_output] = brightness
    return lensed_galaxy

lensed_galaxy = create_lensed_image(source_positions,galaxy ,x_min , y_min)


#visualising
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(galaxy , origin ="lower")
plt.title("Original Source")

plt.subplot(1,2,2)
plt.imshow(lensed_galaxy , origin ="lower")
plt.title("Distributed-Mass Lensed Source")

plt.show()
      

          

    


            








