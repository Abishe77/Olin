import numpy as np
import matplotlib.pyplot as plt
step =0.5
point = 10
def create_galaxy():
    coordinates =[]
    for i in np.arange(-point , point +1 , step):
        for j in np.arange(-point , point+1 , step):
                coordinates.append([i,j])

    return coordinates

def calculate_dist():
     grid = create_galaxy()
     result={}
     for i in range(len(grid)):
          r = np.sqrt((grid[i][0] - 0)**2 + (grid[i][1]-0)**2)
          result[tuple(grid[i])] = r
     return result

'''
What we did is create a square grid with x,y coordinates
then calculated distance from (0,0) to every other (x,y)
inside the grid

Our net challenge is to turn those pixels i.e, (x,y) into
brightness value , like center has max brightness as
we move away from centre , i.e if dist we calculated
increases then the brightness should gradually fade
, so we need a smooth decaying function , what we use
here is gaussian decay function

'''

def gaussian_decay(A =1 , std = 2.5):
     map = calculate_dist()
     b_map = {}
     for keys in map :
          brightness = A * np.exp(- map[keys]**2 / (2 * std**2))
          b_map[keys] = brightness

     return b_map


def see_galaxy():
    brightness_map = gaussian_decay()
    x_values = []
    y_values =[]
    for x,y in brightness_map:
         x_values.append(x)
         y_values.append(y)
    min_x = min(x_values) #For numpy's arrays indexing
    min_y = min(y_values)
    max_x = max(x_values)
    max_y = max(y_values)
    #To set dimensions for the array
    width = int(((max_x - min_x) / step) + 1)
    height = int(((max_y - min_y) / step) + 1)
    galaxy = np.zeros((width ,height))

    for x , y in brightness_map:
         index_x =int((x - min_x)/step)
         index_y = int((y-min_y)/step)
         galaxy[index_x][index_y] = brightness_map[(x,y)]

    return galaxy , min_x , min_y 
#returns the actual brightness of pixels

def visualize_galaxy(galaxy):
     plt.imshow(galaxy)
     plt.show() #takes the pixelated grids and turns it to visuals

'''
Now going to map i.e, using the galaxy we generated
we ask a question "how will this look like if I put a 
lens in-between?" so what we do is using the image grid
we place is over the lens image and map out the estimated
brightness of how the source could have produced
'''

def interpolate(galaxy , x ,y):
     x0 = int(np.floor(x))
     x1 = x0 + 1

     y0 = int(np.floor(y)) #because of -ve value crisis
     y1 = y0 + 1
     #suppose some coordinates are not inside source grid
     #we use "bilinear interpolation" to solve that

     fx = x - x0
     fy = y - y0

     q00 = galaxy[x0,y0]
     q10 = galaxy[x1,y0]
     q01 = galaxy[x0,y1]
     q11 = galaxy[x1,y1]

     #interpolation
     r0 = (1-fx) * q00 + fx * q10 #across x
     r1 = (1-fx) * q01 + fx * q11

     #across y
     brightness = (1-fy) * r0 + fy * r1
     return brightness


#Actual source coordinate mapper
def lens_map(x,y,theta_e):
     theta_sq = x**2 + y**2 #2D angular position vector
     if theta_sq != 0:
      alpha_x = (theta_e**2 * x) / theta_sq
      alpha_y = (theta_e**2 * y) / theta_sq
      beta_x = x - alpha_x
      beta_y = y - alpha_y
      return beta_x , beta_y
     else:
          return None
     
#how much magnitude affects the coordinate 
#is what aplha gives and alpha x is
#how much x is affected and same goes for 
#y respectively
'''Lens eq is beta = theta - alpha 
where theta = (x,y) an vector
alpha = (x,y) (theta_e^2 / theta_sq)
and alpha is lens induced angular deflection
vector and alpha_x , alpha_y gives
respective deflection wrt coordinates'''

#connecting the whole pipeline
'''
galaxy , min_x , min_y  = see_galaxy()
lensed_galaxy = np.zeros_like(galaxy)
for x in np.arange(-point , point+1 , step):
     for y in np.arange(-point , point +1 , step):
          mapped = lens_map(x,y,theta_e=5)

          if mapped is None:
               continue

          beta_x , beta_y = mapped
          # Convert physical source coordinates
        # into source-array coordinates
          index_x = (beta_x - min_x) / step
          index_y = (beta_y - min_y) / step
          #Check whether mapped source coordinate
          #lies inside our finite source image

          if(
               index_x < 0
               or index_x >= galaxy.shape[0] -1
               or index_y < 0
               or index_y >= galaxy.shape[1] -1
          ):
               brightness =0
          else:
               brightness = interpolate(galaxy , index_x,index_y)

          #Convert physical image coordinate
          # #(x,y) into o/p array coordinates
          output_x = int((x - min_x) / step)
          output_y = int((y - min_y) / step)

          lensed_galaxy[output_x, output_y] = brightness 


#visulaize
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(galaxy, origin="lower")
plt.title("Original Source")

plt.subplot(1, 2, 2)
plt.imshow(lensed_galaxy, origin="lower")
plt.title("Lensed Source")

plt.show()                  
'''
          




     

             
          



     



