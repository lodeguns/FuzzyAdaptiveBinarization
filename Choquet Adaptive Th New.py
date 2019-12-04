#!/usr/bin/env python
# coding: utf-8

# In[45]:


# Choquet adaptive thresholding: two step algorithm
import progressbar
from time import sleep
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
from skimage import measure   
from pynverse import inversefunc
import time
import scipy.misc
get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import warnings
import numpy as np
#Otsu trhesholding
from skimage import data
from skimage import filters
from skimage import exposure

#format the output in a readable format
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(precision=0,formatter={'float_kind':float_formatter})


if not sys.warnoptions:
    warnings.simplefilter("ignore")
    


# In[46]:


#function section
### import img
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def plot_it(img):
    plt.figure(figsize = [8,8])
    arr = np.asarray(img)
    plt.imshow(arr, cmap='gray', vmin=0, vmax=arr.max())
    plt.title(namestr(img, globals()))
    plt.show()
    
def import_img(img_path):
    img = cv2.imread(img_path, 0)
    img_reverted= cv2.bitwise_not(img)
    norm_img = img_reverted / 255.0
    #plot_it(norm_img)
    print(norm_img)
    print(norm_img.shape)
    print(norm_img.size)
    return(norm_img)

### cumulative G function (sum-table algorithm)

def compute_summed_area_table(image):
    # image is a 2-dimensional array containing ints or floats, with at least 1 element.
    height = len(image)
    width = len(image[0])
    new_image = [[0.0] * width for _ in range(height)] # Create an empty summed area table
    for row in range(0, height):
        for col in range(0, width):
            if (row > 0) and (col > 0):
                new_image[row][col] = image[row][col] +                     new_image[row][col - 1] + new_image[row - 1][col] -                     new_image[row - 1][col - 1]
            elif row > 0:
                new_image[row][col] = image[row][col] + new_image[row - 1][col]
            elif col > 0:
                new_image[row][col] = image[row][col] + new_image[row][col - 1]
            else:
                new_image[row][col] = image[row][col]
    return new_image


def get_int_img_m1(input_img):
    h, w = input_img.shape
    #integral img
    int_img = np.zeros_like(input_img, dtype=np.uint32)
    for col in range(w):
        for row in range(h):
            int_img[row,col] = input_img[0:row+1,0:col+1].sum()
    return int_img

def cdf_image(input_img):
    nh, binn = np.histogram(input_img)
    cdf = np.cumsum(nh)
    return([cdf, nh, binn])


# In[ ]:





# In[47]:


# Adaptive choquet 
# OPT= 0 Hamacher
# OPT= 1 Discrete Choquet
# Opt= 2 Discrete Choquet with F1,F2 on the distributive property

def compute_choquet(choquet_order, fuzzy_mu, opt=0):
    C=0
    if opt==0:   # Choquet Hamacher
        for i in range(len(choquet_order)-1):
            j = i +1
            C = C + (choquet_order[j] * fuzzy_mu[i])/(choquet_order[j] + fuzzy_mu[i] - (choquet_order[j] * fuzzy_mu[i]))
    if opt==1:  #Choquet
        for i in range(len(choquet_order)-1):
            j = i +1
            C = C + ((choquet_order[j] - choquet_order[j-1] )*fuzzy_mu[i])
    if opt ==2: #Choquet F1 F2 
        for i in range(len(choquet_order)-1):
            j = i +1
            C = C + (np.sqrt(choquet_order[j]*fuzzy_mu[i]) -  max( (choquet_order[j]+fuzzy_mu[i] -1) , 0))
    return(C)



def compute_sugeno(sugeno_order, fuzzy_mu):
    S = np.empty((1), float)

    for i in range(len(sugeno_order)):
        S = np.append(S, min(sugeno_order[i], fuzzy_mu[i]))
        #print(S)
        #print('sugeno: ' + str(choquet_order[j]) + " " + str(fuzzy_mu[i]) + " " + str(max(S)))
    return(max(S))    



## Integral Choquet and Sugeno image.
def adaptive_choquet_itegral(input_img, int_img, opt,log=False):
    
    h, w = input_img.shape
    th_mat = np.zeros(input_img.shape)
    choquet_mat =  np.zeros(input_img.shape)
    sugeno_mat =  np.zeros(input_img.shape)
    count_matrix =  np.zeros(input_img.shape)
    
    for col in range(w):       #i
        for row in range(h):   #j
            #SxS region

            y0 = int(max(row-1, 0))
            y1 = int(min(row, h-1))
            x0 = int(max(col-1, 0))
            x1 = int(min(col, w-1))
                      
            count = (y1-y0)*(x1-x0)
            count_matrix[row, col] = count
            choquet_order = -1    
            sum_ = -1
            fuzzy_mu = -1
            if count == 0:
                if x0 == x1 and y0 == y1:
                    sum_ = int_img[y0, x0]
                    C_   = sum_
                    S_   = sum_
                if x1 == x0 and y0 != y1:
                    sum_ = (int_img[y1, x1] + int_img[y0, x1])/2
                    choquet_order = np.asarray([0,int_img[y0, x1], int_img[y1, x1]]) 
                    sugeno_order  = np.asarray([int_img[y0, x1], int_img[y1, x1]]) 
                    fuzzy_mu = np.asarray([1, 0.5])
                    C_ = compute_choquet(choquet_order, fuzzy_mu,opt)
                    S_  = compute_sugeno(sugeno_order, fuzzy_mu)
                if y1 == y0 and x1 != x0:
                    sum_ = (int_img[y1, x1] + int_img[y1, x0])/2
                    choquet_order = np.asarray([0,int_img[y1, x0], int_img[y1, x1]])
                    sugeno_order  = np.asarray([int_img[y1, x0], int_img[y1, x1]])
                    fuzzy_mu = np.asarray([1, 0.5])
                    C_ = compute_choquet(choquet_order, fuzzy_mu,opt)
                    S_  = compute_sugeno(sugeno_order, fuzzy_mu)
            else:
                sum_ = int_img[y1, x1] - int_img[y0, x1] - int_img[y1, x0] + int_img[y0, x0]
                if(int_img[y0, x1] > int_img[y1, x0] ):
                    choquet_order = np.asarray([0,int_img[y0, x0], int_img[y1, x0], int_img[y0, x1], int_img[y1, x1]])
                    sugeno_order = np.asarray([int_img[y0, x0], int_img[y1, x0], int_img[y0, x1], int_img[y1, x1]])
                else:
                    choquet_order = np.asarray([0,int_img[y0, x0], int_img[y0, x1], int_img[y1, x0], int_img[y1, x1]])
                    sugeno_order  = np.asarray([int_img[y0, x0], int_img[y0, x1], int_img[y1, x0], int_img[y1, x1]])

                fuzzy_mu = np.asarray([1, 0.75, 0.50, 0.25])
                C_ = compute_choquet(choquet_order, fuzzy_mu,opt)
                S_  = compute_sugeno(sugeno_order, fuzzy_mu)
                
            th_mat[row,col]      = sum_ 
            choquet_mat[row,col] = C_   
            sugeno_mat[row,col]  = S_     

            if(log):
                coords_window = np.zeros_like(input_img)

                #coords_window[x0:x1,y0:y1] = 1.0
                coords_window[y0, x0] = 0.2
                coords_window[y1, x0] = 0.4
                coords_window[y0, x1] = 0.6
                coords_window[y1, x1] = 0.8
                plot_it(coords_window)
                
                print("Search_region")
                print("x0:" + str(x0) + " x1:"+ str(x1) + " y0:" + str(y0) + " y1:" + str(y1) )
                print("Row:" + str(row) + " Col:" + str(col))
                print("Count: " + str(count))
                print("choquet fixed ordered and fuzzy mu")            
                print(choquet_order)
                print(fuzzy_mu)
                print("choquet calculus")
                print(C_)
                print("sugeno calculus")
                print(S_)
                print("Input mat")
                print(input_img)
                print("Int img")
                print(int_img)
                print("I integral mat: ")
                print(th_mat)
                print("C_ choquet")
                print(choquet_mat)
                print("S_ sugeno")
                print(sugeno_mat)
                print("Count matrix")
                print(count_matrix)
                print("-------")

    return  choquet_mat, sugeno_mat, count_matrix
 


# In[ ]:





# In[48]:


## Classic Bradley Apprroach
def adaptive_thresh(input_img, int_img, a1=8, a2=2, T=0.15):     
    out_img = np.zeros_like(input_img) 
    h, w = input_img.shape
    S = w/a1
    s2 = S/a2
    th_mat = np.zeros(input_img.shape)
    for col in range(w):
        for row in range(h):
            #SxS region
            y0 = int(max(row-s2, 0))
            y1 = int(min(row+s2, h-1))
            x0 = int(max(col-s2, 0))
            x1 = int(min(col+s2, w-1))

            count = (y1-y0)*(x1-x0)
            sum_ = int_img[y1, x1] - int_img[y0, x1] - int_img[y1, x0] + int_img[y0, x0]

            th_mat[row,col] = sum_/count
            
            if input_img[row, col]*count < sum_*(1.-T)/1.:
                out_img[row,col] = 0
            else:
                out_img[row,col] = 1

    return np.asarray(out_img), th_mat




#Novel choquet adaptive approach
def adaptive_thresh2(input_img, int_img, a1=4, a2=1, T=0, log=False):
    if T==0:
        T = filters.threshold_otsu(input_img)
        T = T

    out_img_choquet = np.zeros_like(input_img) 
    out_img_sugeno  = np.zeros_like(input_img)
    choquet_mat     = np.zeros_like(input_img)
    sugeno_mat      = np.zeros_like(input_img)
    h, w = input_img.shape
    S = w/a1
    s2 = S/a2


    for col in range(w):
        for row in range(h):
            y0 = int(max(row-s2, 0))
            y1 = int(min(row+s2, h-1))
            x0 = int(max(col-s2, 0))
            x1 = int(min(col+s2, w-1))
            count = (y1-y0)*(x1-x0)   
            sum_ = -1
            fuzzy_mu = -1
            if count == 0:
                if x0 == x1 and y0 == y1:
                    sum_ = int_img[y0, x0]
                    S_   = sum_
                if x1 == x0 and y0 != y1:
                    sum_ = int_img[y1, x1] - int_img[y0, x1]
                    sugeno_order  = np.asarray([int_img[y0, x1], int_img[y1, x1]]) 
                    fuzzy_mu = np.asarray([1, 0.5])

                    S_  = compute_sugeno(sugeno_order, fuzzy_mu)
                if y1 == y0 and x1 != x0:
                    sum_ = int_img[y1, x1] - int_img[y1, x0]
                    sugeno_order  = np.asarray([int_img[y1, x0], int_img[y1, x1]])
                    fuzzy_mu = np.asarray([1, 0.5])

                    S_  = compute_sugeno(sugeno_order, fuzzy_mu)
            else:
                sum_ = int_img[y1, x1] - int_img[y0, x1] - int_img[y1, x0] + int_img[y0, x0]
                if(int_img[y0, x1] > int_img[y1, x0] ):
                     sugeno_order = np.asarray([int_img[y0, x0], int_img[y1, x0], int_img[y0, x1], int_img[y1, x1]])
                else:
                     sugeno_order  = np.asarray([int_img[y0, x0], int_img[y0, x1], int_img[y1, x0], int_img[y1, x1]])
                fuzzy_mu = np.asarray([1, 0.75, 0.50, 0.25])
                S_  = compute_sugeno(sugeno_order, fuzzy_mu)
           
            
            choquet_mat[row,col] = sum_/count

            if input_img[row, col]*count  < sum_ * (1.-T)/1.:
                out_img_choquet[row,col] = 0
            else:
                out_img_choquet[row,col] = 1

            sugeno_mat[row,col] = S_/count
            #note is not only T
            if input_img[row, col]*count   <  S_ * (1.- T)/1.:
                out_img_sugeno[row,col] = 0
            else:
                out_img_sugeno[row,col] = 1

    return out_img_choquet, out_img_sugeno, choquet_mat, sugeno_mat, T




# In[ ]:


#Qualitative comparisons


# Compute the mean squared error and structural similarity
# index for the images
def compare_images(img1, img2):
    m = mse(img1, img2)
    s = measure.compare_ssim(img1, img2,  data_range=img2.max() - img2.min(), multichannel=False)
    ret = np.array([m,s])
    #the higher the ssim, the more "similar"
    return(ret)

def mse(img1, img2):
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img2.shape[1])
    #the lower the error, the more "similar"
    return(err)


#simple listing class in order to collect the results
class results_collector(object):
    def __init__(self, name, original_img, choquet_mat, sugeno_mat, count_matrix, 
                 out_img_adapt_choquet,out_img_sugeno,out_img_bradley, c_m, s_m, T, elapsed_time,
                 mse_choquet, mse_sugeno, mse_bradley, ssim_choquet, ssim_sugeno, ssim_bradley,
                 th, a1, a2):
        self.name = name, 
        self.img = original_img,
        self.choquet_mat = choquet_mat,  
        self.sugeno_mat = sugeno_mat,
        self.count_matrix = count_matrix,
        self.out_img_adapt_choquet = out_img_adapt_choquet,
        self.out_img_sugeno = out_img_sugeno,
        self.out_img_bradley = out_img_bradley
        self.c_m = c_m,
        self.s_m = s_m,
        self.T  = T,                                                
        self.elapsed_time = elapsed_time,
        self.mse_choquet  = mse_choquet, 
        self.mse_sugeno   = mse_sugeno, 
        self.mse_bradley  = mse_bradley, 
        self.ssim_choquet = ssim_choquet, 
        self.ssim_sugeno  = ssim_sugeno,
        self.ssim_bradley = ssim_bradley,
        self.th = th,
        self.a1 = a1, 
        self.a2 = a2
        
        
#Embedded method for comparisons between groundtruth and Choquet thresholded images
def compute_multi_thresh(test_images, gt_images, opt = 0, T=0, a1=2, a2=2):
    count=0
    resc = []
    for i in test_images:
        test_image  =  i
        #plot_it(test_image)

        S1 = np.asarray(compute_summed_area_table(test_image))
        #S1 = get_int_img_m1(test_image)
        choquet_mat, sugeno_mat, count_matrix =                              adaptive_choquet_itegral(np.asarray(test_image),
                            S1, 
                            opt,        
                            log=False )   
        #Choquet Adaptive Thresh
        out_img_adapt_choquet, out_img_sugeno, c_m, s_m, T =          adaptive_thresh2(np.asarray(test_image),
                            np.asarray(choquet_mat), 
                            a1 = a1,
                            a2 = a2,
                            T= T,
                            log=False )     #with compute_summed_area table doesn't work.


        #Bradley Adaptive Thresh
        S1 = get_int_img_m1(test_image)
        out_img_bradley, bradley_int_mat =          adaptive_thresh(np.asarray(test_image),
                        S1 , 
                        a1=a1,
                        a2=a2,
                        T=T) 
    
        #compare it
        mse_choquet, ssim_choquet = compare_images(gt_images[count], out_img_adapt_choquet)
        mse_sugeno,  ssim_sugeno  = compare_images(gt_images[count], out_img_sugeno)
        mse_bradley, ssim_bradley = compare_images(gt_images[count], out_img_bradley)
    
        #

        resc.append(results_collector("Comparisons", i, choquet_mat,  
                                                    sugeno_mat,count_matrix,
                                                    out_img_adapt_choquet,
                                                    out_img_sugeno,
                                                    out_img_bradley,
                                                    c_m,
                                                    s_m,
                                                    T,                                                
                                                    elapsed_time,
                                                    mse_choquet,  
                                                    mse_sugeno,
                                                    mse_bradley,
                                                    ssim_choquet,
                                                    ssim_sugeno,
                                                    ssim_bradley,
                                                    T,
                                                    a1, 
                                                    a2))

        
        count += 1
        
    return(resc)        


def add_random_noise(small_image, perc=1):
    np.random.seed(1)
    mu, sigma = 0, 1 # mean and standard deviation
    s = np.random.normal(mu, sigma, small_image.shape)
    img_n = np.abs(s/s.max()) * perc
    img_ret = small_image + img_n
    return(img_ret)


# In[82]:


### Testing Grad/Glaze images vs Groundtruth / GT noise vs GT / Test+noise vs GT

def test_exp(test_images, gt_images, a1=7, a2=7, opt=0, scale = 0.01, noise_gt = -1, noise_test=-1):
    resc_a = []
    elapsed_time=0
    
    ### Add noise on the GroundTruth
    if noise_gt > 0:
        noise_img = []
        for i in range(len(gt_images)):
            noise_img.append(add_random_noise(gt_images[i], noise_gt))
        test_images = noise_img
        
    #Add noise on the test images
    if noise_test > 0:
        noise_img = []
        for i in range(len(test_images)):
            noise_img.append(add_random_noise(test_images[i], noise_test))
        test_images = noise_img
    # Test test_images or noised ones with respect the GT.
    for i in range(a1):
        for j in range(a2):
            x = scale
            if(i >= j ):
                print("Testing image conf ( i: " + str(i) + " j: " + str(j) + ")")
                t1 = time.process_time()
                while(x <= 1.01):
                    resc = compute_multi_thresh(test_images,gt_images,
                                    opt = opt,
                                    T=x, 
                                    a1=i+1, 
                                    a2=j+1)
                    x = x + scale
                    resc_a.append(resc)
                elapsed_time = time.process_time() - t1
                print('Out: {} images processed in {} seconds'.format(str(len(resc)), round(elapsed_time ,3)))
    return(resc_a)


##  Simple testing prints
##  It should return the list of the stuff
def search_results(resc_b, ssim_th = 0.5, attention_image = -1):
    count=0
    for i in range(len(resc_b)):
        for j in range(len(resc_b[-1])):
            if(resc_b[i][j].ssim_choquet[0] > resc_b[i][j].ssim_bradley[0] 
                and resc_b[i][j].ssim_choquet[0] > ssim_th
                and resc_b[i][j].a1[0] != resc_b[i][j].a2):
                count= count+1
                print('{}-th image -------------------\n mse: C {} S {} B {}, \nssid: C {} S {} B {} \na1: {}, a2: {}, th: {}'.format(
                str(j), 
                round(resc_b[i][j].mse_choquet[0],3),
                round(resc_b[i][j].mse_sugeno[0],3),
                round(resc_b[i][j].mse_bradley[0],3),
                round(resc_b[i][j].ssim_choquet[0],3),
                round(resc_b[i][j].ssim_sugeno[0],3),
                round(resc_b[i][j].ssim_bradley[0],3),
                str(resc_b[i][j].a1[0]),
                str(resc_b[i][j].a2),
                round(resc_b[i][j].th[0], 4) ))
                if(attention_image >= 0):
                    if(j==attention_image):
                        print("**********************************************************************************")
    print("Percentage of coverage around all the possible configurations" + str(count/(len(resc_b)*len(resc_b[-1]))))
    


# In[83]:


################################################################################
#### Test on a single image:
################################################################################

small_image  =  1.0 - import_img('./original/00.bmp')
plot_it(small_image)
S1 = np.asarray(compute_summed_area_table(small_image))
cdf_img   = cdf_image(small_image)
int_img   = get_int_img_m1(small_image)   # common 
#int_img2  = get_int_img_m2(small_image, cum_distr)   #choquet int img

print("Image")
print(np.asarray(small_image))

print("summed area table")
print(np.asarray(summ_at))

print("integral image")
print(int_img)


plt.plot(np.asarray(cdf_img[0]), np.asarray( cdf_img[2][0:len(cdf_img[2])-1]), 'r--')

print("cumulative distribution of the image")
print(np.asarray(cdf_img[0]))

print("histogram")
print(np.asarray(cdf_img[1]))

print("range values")
print(np.asarray(cdf_img[2]))


choquet_mat, sugeno_mat, count_matrix =                      adaptive_choquet_itegral(np.asarray(small_image),
                    S1, 
                    1,
                    log=False )     
print("C mat")
plot_it(choquet_mat)

print("S mat")
plot_it(sugeno_mat)



print("-----------------------------------------------------------------------------------")
#Otsu T parameter
print("Image thresholded with the choquet integral image and an automatic Otsu threshold")
out_img_adapt_choquet, out_img_sugeno, c_m, s_m, T =  adaptive_thresh2(np.asarray(small_image),
                    np.asarray(choquet_mat), 
                    a1 = 16,
                    a2 = 2, #Leave T = 0 for the Otsu
                    log=False )     #con compute_summed_area table doesn't work.

print("Threshold  " + str(T))
plot_it(out_img_adapt_choquet)
plot_it(out_img_sugeno)
plot_it(c_m)
plot_it(s_m)


print("-----------------------------------------------------------------------------------")
#Manual Parameter
print("Image thresholded with the choquet integral image and a fixed manual threshold.")
out_img_adapt_choquet, out_img_sugeno, c_m, s_m, T =  adaptive_thresh2(np.asarray(small_image),
                    np.asarray(choquet_mat), 
                    a1 = 16,
                    a2 = 2,
                    T  = 0.2,
                    log=False )     #con compute_summed_area table doesn't work.

print("Threshold  " + str(T))
plot_it(out_img_adapt_choquet)
plot_it(out_img_sugeno)
plot_it(c_m)
plot_it(s_m)



# In[84]:


################################################################################
#### Toy dataset # Testing complex gradients, glazes, additive noise, smoothness
################################################################################

#Prepare the list data structures
#Groundtruth images
gt_images        = []
# Smothed, glazed GT images
test_images      = []



# In[85]:


###
# Definition of the toy dataset
### 
small_image1 = [[0, 0, 0, 0, 0, 0,0,0,0], 
                [0, 1, 0, 1, 0, 0,0,0,0], 
                [1, 1, 1, 1, 1, 0,0,0,0], 
                [0, 1, 0, 1, 0, 0,0,0,0],
                [0, 0, 1, 1, 1, 0,0,0,0],
                [0, 0, 0, 1, 0, 0,0,0,0],
                [0, 0, 1, 1, 1, 0,0,0,0],
                [0, 0, 0, 1, 0, 0,0,0,0],
                [0, 0, 1, 1, 1, 0,0,0,0]]



small_image1 = np.asarray(small_image1, dtype="float32")
gt_images.append(small_image1)
plot_it(small_image1)

small_image1 = [[0.2, 0.2, 0.1, 0.2, 0.15, 0.14,0.13,0.12,0.11], 
                [0.16, 0.6, 0.2, 0.3, 0.15, 0.14,0.13,0.12,0.11], 
                [0.6, 0.5, 0.6, 0.7, 0.8, 0.14,0.13,0.12,0.11], 
                [0.14, 0.5, 0.2, 0.3, 0.15, 0.14,0.13,0.12,0.11],
                [0.15, 0.12, 0.3, 0.4, 0.3, 0.14,0.13,0.12,0.11],
                [0.14, 0.13, 0.2, 0.4, 0.15, 0.14,0.13,0.12,0.11],
                [0.15, 0.12, 0.3, 0.3, 0.3, 0.14,0.13,0.12,0.11],
                [0.14, 0.13, 0.2, 0.26, 0.1, 0.14,0.13,0.12,0.11],
                [0.15, 0.12, 0.25, 0.25, 0.25, 0.14,0.13,0.12,0.11]]

small_image1 = np.asarray(small_image1, dtype="float32")
test_images.append(small_image1)
plot_it(small_image1)

small_image2 = [[0, 0, 0, 0, 1,0,0,0,0], 
                [0, 0, 0, 1, 0,1,0,0,0], 
                [0, 0, 1, 0, 0,0,1,0,0], 
                [0, 1, 0, 0, 0,0,0,1,0],
                [0, 1, 0, 0, 0,0,0,1,0],
                [0, 0, 1, 0, 0,0,1,0,0], 
                [0, 0, 0, 1, 0,1,0,0,0],
                [0, 0, 0, 0, 1,0,0,0,0]]

small_image2 = np.asarray(small_image2, dtype="float32")
gt_images.append(small_image2)
plot_it(small_image2)

small_image2 = [[0.22, 0.19, 0.19, 0.18, 0.5, 0.11, 0.08, 0.06,0.02], 
                [0.22, 0.19, 0.19, 0.5,  0.15,0.6,  0.08, 0.06,0.02], 
                [0.22, 0.19, 0.40, 0.18, 0.15,0.11, 0.7,  0.06,0.02], 
                [0.22, 0.30, 0.19, 0.18, 0.15,0.11, 0.08, 0.8,0.02],
                [0.22, 0.30, 0.19, 0.18, 0.15,0.11, 0.08, 0.8,0.02],
                [0.22, 0.19, 0.40, 0.18, 0.15,0.11, 0.7,  0.06,0.02], 
                [0.22, 0.19, 0.19, 0.5,  0.15,0.6,  0.08, 0.06,0.02],
                [0.22, 0.19, 0.19, 0.18, 1   ,0.11, 0.08, 0.06,0.02]]

small_image2 = np.asarray(small_image2, dtype="float32")
plot_it(small_image2)
test_images.append(small_image2)


small_image3 = [[0,0,0, 0,  0, 0, 0,0],
                [0,0,0, 0,  0, 0, 0,0],
                [0,0,0, 1,  0, 0, 0,0], 
                [0,0,1, 1,  1, 0, 0,0], 
                [0,0,0, 1,  0, 1, 0,0], 
                [0,0,0, 1,  1, 1, 1,0],
                [0,0,0, 1,  0, 1, 0,0],
                [0,0,0, 1,  0, 0, 0,0]]
 
small_image3 = np.asarray(small_image3, dtype="float32")
plot_it(small_image3)
gt_images.append(small_image3)



small_image3 = [[0.18,0.22, 0.15, 0.22,   0.20,   0.17, 0.15,0.14], 
                [0.18,0.22, 0.15, 0.22,   0.20,   0.15, 0.17,0.1], 
                [0.18,0.22, 0.15, 0.45,   0.20,   0.17, 0.15,0.14], 
                [0.17,0.21, 0.35, 0.45,   0.55,   0.15, 0.17,0.1], 
                [0.17,0.20, 0.15, 0.45,   0.20,   0.65, 0.15,0.14], 
                [0.18,0.21, 0.15, 0.45,   0.55,   0.65, 0.75,0.1],
                [0.19,0.22, 0.15, 0.45,   0.20,   0.65, 0.15,0.14],
                [0.18,0.22, 0.15, 0.35,   0.20,   0.15, 0.17,0.1]]

small_image3 = np.asarray(small_image3, dtype="float32")
plot_it(small_image3)
test_images.append(small_image3)

small_image4 = [[0, 0,   0,   0, 0,0,0,0], 
                [0, 0,   0,   0, 1,1,1,0], 
                [0, 0,   0,   1, 1,1,1,0], 
                [0, 0,   1,   1, 1,1,1,0], 
                [0, 1,   1,   1, 1,1,1,0], 
                [0, 0,   1,   1, 1,1,1,0],
                [0, 0,   0,   1, 1,1,1,0],
                [0, 0,   0,   0, 1,1,1,0] 
               ]
 
small_image4 = np.asarray(small_image4, dtype="float32")
small_image6 = np.asarray(np.transpose(small_image4), dtype="float32")
plot_it(small_image4)
gt_images.append(small_image4)
gt_images.append(small_image6)

small_image4 = [[0.1, 0.1,   0.3,   0.2, 0.2,0.1,0,  0], 
                [0.1, 0.15,   0.3,   0.2, 0.4,0.6,0.6,0], 
                [0.1, 0.15,   0.3,   0.5, 0.5,0.5,0.6,0], 
                [0.2, 0.15,   0.6,   0.5, 0.55,0.5,0.6,0], 
                [0.2, 0.8,   0.7,   0.5, 0.55,0.5,0.6,0], 
                [0.2, 0.15,   0.6,   0.5, 0.55,0.5,0.6,0],
                [0.1, 0.1,   0.3,   0.5, 0.5,0.5,0.6,0],
                [0.1, 0.1,   0.3,   0.2, 0.5,0.5,0.6,0] 
               ]


small_image4 = np.asarray(small_image4, dtype="float32")
small_image6 = np.asarray(np.transpose(small_image4), dtype="float32")
plot_it(small_image4)
test_images.append(small_image4)
test_images.append(small_image6)

small_image5 = [[1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0., 0., 0.], 
                [0., 0., 1., 0., 0., 0., 0., 0.], 
                [0., 0., 0., 0., 1., 0,  0., 0.], 
                [0., 0., 0., 1., 1., 1., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.]
               ]

small_image5 = np.asarray(small_image5, dtype="float32")
plot_it(small_image5)
gt_images.append(small_image5)


small_image5 = [[0.4, 0. , 0.,   0., 0., 0., 0., 0.],
                [0. , 0.5, 0.,   0., 0., 0., 0., 0.], 
                [0. , 0. , 0.6, 0.1, 0.1, 0.1, 0.1, 0.], 
                [0. , 0. , 0.1, 0.1, 1.0, 0.1, 0.1, 0.], 
                [0. , 0. , 0.1, 0.7, 0.8, 0.9, 0.1, 0.], 
                [0. , 0. , 0.1, 0.1, 0.9, 0.1, 0.1, 0.],
                [0. , 0. , 0.1, 0.2, 0.1, 0.1, 0.1, 0.],
                [0. , 0. ,  0., 0., 0., 0., 0., 0.]
               
               ]

small_image5 = np.asarray(small_image5, dtype="float32")
plot_it(small_image5)
test_images.append(small_image5)

small_image7 = [[0,0,  0, 0,   0, 0,0,0], 
                [0,0,  0, 0,   0, 0,0,0], 
                [0,0,  0, 1,   0, 0,0,0], 
                [0,0,  1, 0  , 1, 0,0,0], 
                [0,0,  0, 1,   0, 0,0,0], 
                [0,0,  1, 0,   1, 0,0,0],
                [0,0,  0, 1,   0, 0,0,0],
                [0,0,  0, 0,   0, 0,0,0], ]

small_image7 = np.asarray(small_image7, dtype="float32")
plot_it(small_image7)
gt_images.append(small_image7)


small_image7 = [[0,0.1, 0.2, 0.2,   0.2, 0.1,0,0], 
                [0,0.1, 0.2, 0.2,   0.2, 0.1,0,0], 
                [0,0.1, 0.2, 0.7,   0.2, 0.1,0,0], 
                [0,0.3,  0.6, 0.2  , 0.6, 0.1,0,0], 
                [0,0.3,  0.2, 0.8,   0.2, 0.25,0,0], 
                [0,0.1,  0.7, 0.2,   0.6, 0.1,0,0],
                [0,0.1,  0.1, 0.6,   0.2, 0.2,0,0],
                [0,0.1,  0.1, 0.1,   0.2, 0.2,0,0]
                ]

small_image7 = np.asarray(small_image7, dtype="float32")
plot_it(small_image7)
test_images.append(small_image7)

small_image8 = [
               [0, 0, 0,  0,  0,0,1,0],
               [0, 0, 0,  0,  0,1,0,0],
               [0, 0, 0,  0,  1,0,0,0], 
               [ 0, 0, 0,  1, 0,0,0,0], 
               [ 0, 0, 1,  0, 0,0,0,0], 
               [ 0, 1, 0,  0, 0,0,0,0],
               [ 1, 0, 0,  0, 0,0,0,0],
               [0, 0, 0, 0, 0, 0, 0, 0]]

small_image8 = np.asarray(small_image8, dtype="float32")
plot_it(small_image8)
gt_images.append(small_image8)

small_image8 = [[0,  0,  0,   0,  0,        0.4,  1, 0.5],
               [0,  0,      0,   0,   0.3,  0.95,  0.4, 0.5],
               [0,  0,       0, 0.3,  0.9,  0.4,  0,   0], 
               [0,  0,     0.3, 0.8,  0.3,  0,    0,   0], 
               [0,   0.2,  0.8, 0.3,  0,    0,    0,   0], 
               [0.2, 0.8,  0.3,  0,  0,     0,    0,   0],
               [0.8, 0.2,  0,    0,  0,     0,    0,   0],
               [0.2,   0,  0,    0,  0,     0,    0,   0]]

small_image8 = np.asarray(small_image8, dtype="float32")
plot_it(small_image8)
test_images.append(small_image8)


# In[86]:



### OPT 0,1,2 on Testing imgs vs GT
test_0a = test_exp(test_images, gt_images, a1=7, a2=7, opt=0, scale = 0.2)
test_0b = test_exp(test_images, gt_images, a1=7, a2=7, opt=1, scale = 0.2)
test_0c = test_exp(test_images, gt_images, a1=7, a2=7, opt=2, scale = 0.2)

search_results(test_0a, ssim_th = 0.3, attention_image = 2)
search_results(test_0b, ssim_th = 0.3, attention_image = 2)
search_results(test_0c, ssim_th = 0.3, attention_image = 2)

### OPT 0,1,2 on GT noised vs GT +20%
test_1a = test_exp(test_images, gt_images, a1=7, a2=7, opt=0, scale = 0.2, noise_gt = 0.2)
test_1b = test_exp(test_images, gt_images, a1=7, a2=7, opt=1, scale = 0.2, noise_gt = 0.2)
test_1c = test_exp(test_images, gt_images, a1=7, a2=7, opt=2, scale = 0.2, noise_gt = 0.2)

search_results(test_1a, ssim_th = 0.3, attention_image = 2)
search_results(test_1b, ssim_th = 0.3, attention_image = 2)
search_results(test_1c, ssim_th = 0.3, attention_image = 2)

### OPT 0,1,2 on Testing imgs noised vs GT + 20%
test_2a = test_exp(test_images, gt_images, a1=7, a2=7, opt=0, scale = 0.2, noise_test = 0.2)
test_2b = test_exp(test_images, gt_images, a1=7, a2=7, opt=1, scale = 0.2, noise_test = 0.2) 
test_2c = test_exp(test_images, gt_images, a1=7, a2=7, opt=2, scale = 0.2, noise_test = 0.2)

search_results(test_2a, ssim_th = 0.3, attention_image = 2)
search_results(test_2b, ssim_th = 0.3, attention_image = 2)
search_results(test_2c, ssim_th = 0.3, attention_image = 2)





# In[23]:


##################### 
### Berkeley Dataset   

imgs =    ['./original/00.bmp', './original/01.bmp','./original/02.bmp',
           './original/03.bmp', './original/04.bmp','./original/05.bmp', 
           './original/06.bmp', './original/07.bmp','./original/08.bmp',
           './original/09.bmp' ]
imgs_gt = [  './gtruth/00.bmp',  './gtruth/01.bmp',  './gtruth/02.bmp',
             './gtruth/03.bmp',  './gtruth/04.bmp',  './gtruth/05.bmp',
             './gtruth/06.bmp',  './gtruth/07.bmp',  './gtruth/08.bmp',
             './gtruth/09.bmp' ]

test_images2  = []
test_images2.append(1.0 - import_img(imgs[0]))
test_images2.append(1.0 - import_img(imgs[1]))
test_images2.append(1.0 - import_img(imgs[2]))
test_images2.append(1.0 - import_img(imgs[3]))
test_images2.append(1.0 - import_img(imgs[4]))
test_images2.append(1.0 - import_img(imgs[5]))
test_images2.append(1.0 - import_img(imgs[6]))
test_images2.append(1.0 - import_img(imgs[7]))
test_images2.append(1.0 - import_img(imgs[8]))
test_images2.append(1.0 - import_img(imgs[9]))


test_images_gt_2  = []
test_images_gt_2.append(1.0 - import_img(imgs_gt[0]))
test_images_gt_2.append(1.0 - import_img(imgs_gt[1]))
test_images_gt_2.append(1.0 - import_img(imgs_gt[2]))
test_images_gt_2.append(1.0 - import_img(imgs_gt[3]))
test_images_gt_2.append(1.0 - import_img(imgs_gt[4]))
test_images_gt_2.append(1.0 - import_img(imgs_gt[5]))
test_images_gt_2.append(1.0 - import_img(imgs_gt[6]))
test_images_gt_2.append(1.0 - import_img(imgs_gt[7]))
test_images_gt_2.append(1.0 - import_img(imgs_gt[8]))
test_images_gt_2.append(1.0 - import_img(imgs_gt[9]))


# In[ ]:


###Berkeley 
### Testing Grad/Glaze images vs Groundtruth - t-norm choquet
### From a range from 0 to 1 it requires 2963.407 seconds
### better fixing a 16/2 and not trying all the possible combs.
brk_resc = []
t1 = time.process_time()
x=0
while(x <= 1.00):
    resc = compute_multi_thresh(test_images2, test_images_gt_2,
                                opt = 0,
                                T=x, 
                                a1=16, 
                                a2=2)
    x = x + 0.01
    brk_resc.append(resc)
    elapsed_time = time.process_time() - t1
    print(x)
    print('Images processed in {} seconds'.format(round(elapsed_time ,3)))
  


# In[110]:


print(len(brk_resc))


# In[114]:


for i in range(len(brk_resc)):
    for j in range(len(brk_resc[-1])):
        if(brk_resc[i][j].ssim_choquet[0] > brk_resc[i][j].ssim_bradley[0] 
            and brk_resc[i][j].ssim_choquet[0] >0.3
            and brk_resc[i][j].a1[0] != brk_resc[i][j].a2):
            count= count+1
            print('{}-th image -------------------\n mse: C {} S {} B {}, \nssid: C {} S {} B {} \na1: {}, a2: {}, th: {}'.format(
            str(j), 
            round(brk_resc[i][j].mse_choquet[0],3),
            round(brk_resc[i][j].mse_sugeno[0],3),
            round(brk_resc[i][j].mse_bradley[0],3),
            round(brk_resc[i][j].ssim_choquet[0],3),
            round(brk_resc[i][j].ssim_sugeno[0],3),
            round(brk_resc[i][j].ssim_bradley[0],3),
            str(brk_resc[i][j].a1[0]),
            str(brk_resc[i][j].a2),
            round(brk_resc[i][j].th[0], 4) ))


# In[176]:


#### Example of the chessboard
test_image = 1.0 - import_img('./original/chessboard.png')

#Choquet Adaptive Thresh
choquet_mat, _, _ =  adaptive_choquet_itegral(np.asarray(test_image),
                            S1,  
                            0,                   #t-norm version
                            log=False )   
out_img_adapt_choquet, _, _, _, T =  adaptive_thresh2(np.asarray(test_image),
                            np.asarray(choquet_mat), 
                            a1 = 16,
                            a2 = 2,
                            T= 0.095,
                            log=False )     #con compute_summed_area table doesn't work.



#Choquet Adaptive Thresh
choquet_mat, _, _ =  adaptive_choquet_itegral(np.asarray(test_image),
                            S1, 
                            1,                  #choquet int version
                            log=False )   
out_img_adapt_choquet2, _, _, _, T =  adaptive_thresh2(np.asarray(test_image),
                            np.asarray(choquet_mat), 
                            a1 = 16,
                            a2 = 2,
                            T= 0.095,
                            log=False )     #con compute_summed_area table doesn't work.


#Bradley Adaptive Thresh
S1 = get_int_img_m1(test_image)
out_img_bradley, bradley_int_mat =  adaptive_thresh(np.asarray(test_image),
                        S1 , 
                        a1=16,
                        a2=2,
                        T=T) 


# In[177]:


#Choquet Adaptive Thresh
plot_it(test_image)
plot_it(out_img_adapt_choquet2)
plot_it(out_img_adapt_choquet)
plot_it(out_img_bradley)

print(compare_images(out_img_adapt_choquet, out_img_bradley))
print(compare_images(out_img_adapt_choquet2, out_img_bradley))


# In[ ]:





# In[ ]:




