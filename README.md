## Adaptive binarization based on fuzzy integrals
This repository contains the manuscript mentioned in the title, and associated code and data sets used for testing our novel methodology. Should you need help running our code, please contact us.

https://link

### Citation

Bardozzo et al.  "Adaptive binarization based on fuzzy integrals" Journal Journal.

### Abstract
**Background:**  	 
Adaptive binarization methodologies thredhold the intensity of the pixels with respect to adjacent pixels exploiting the integral images. In turn, the integral images are generally computed optimally using the summed-area-table algorithm (SAT).                                 
**Results and Conclusions:**   
This document presents a new adaptive binarization technique based on fuzzy integral images through an efficient design of a modified SAT for fuzzy integrals. We define this new methodology as FLAT (Fuzzy Local Adaptive Thresholding). The experimental results show that the proposed methodology have produced an image quality thresholding often better than other traditional or simple neural network models. We propose a new generalization of the Sugeno and CF12 integrals to improve the existing results and how they can be efficiently computed in the Integral Image. Therefore, these new generalized fuzzy integrals can be used as a tool for grayscale processing in real-time and deep-learning applications.			
 
 
**Table S-1**
In the [Table S-1](/theta-dataset) a comparison between our algorithms and the Bradley algorithm on the toy-dataset is provided.

**Visual Examples**

Here, additional visual examples on the [theta-dataset](/theta-dataset) for our 3 methods **A2,A3,A4** with respect 4 traditional binarization methods (note Global Th is the Otsu method):

![alt text](/res0.png)

 
 
## FLAT algortihm (Fuzzy Local Adaptive Thresholding)

Here a whole overview of the FLAT algorithm. More details in the paper. 

![alt text](/image1git.png)


**Source code**: 
The extened tests and implementations are provided in a single Python 3.6.8 script, [here](/fuzzy_adaptive_bin.py).

**Sensitivity and Robustness analysis table for FLAT**: are provided in this [file](sensitivity_robustness)

**Time benchmark on Google Colab**: The JIT optimized implementations of the CF12 - FLAT (the best performing one in terms of SSIM, MSE, Precision and Recall) and the associated benchmarks are analyzed on Google Colab and they are provided in the following link: [link](https://colab.research.google.com/drive/1bdL0tHnW213_AZUHoYX2vq1PhEC75kCA
).

Here, for the impatient, is an implementation of the FLAT methods used [here](/fuzzy_adaptive_bin.py).


**S integral image : Classic summed-area table algorithm (SAT)**

```Python

def compute_summed_area_table(image):
    height = len(image)
    width = len(image[0])
    new_image = [[0.0] * width for _ in range(height)] 
    for row in range(0, height):
        for col in range(0, width):
            if (row > 0) and (col > 0):
                new_image[row][col] = image[row][col] + \
                    new_image[row][col - 1] + new_image[row - 1][col] - \
                    new_image[row - 1][col - 1]
            elif row > 0:
                new_image[row][col] = image[row][col] + new_image[row - 1][col]
            elif col > 0:
                new_image[row][col] = image[row][col] + new_image[row][col - 1]
            else:
                new_image[row][col] = image[row][col]
    return new_image
``` 

**F(A2) Integral image: CF12 - Generalized Sugeno**

```Python
def compute_summed_area_table_F1F2(image ):
    height = len(image)
    width = len(image[0])
    S   = [[0.0] * width for _ in range(height)] 
    S_c = S# Create an empty summed area table
    for row in range(0, height):
        for col in range(0, width):
            if (row > 0) and (col > 0):
                S[row][col] = image[row][col] + S[row][col - 1] +  S[row - 1][col] -  S[row - 1][col - 1]
                if(S[row][col - 1] > S[row - 1][col] ):
                    ov = np.asarray([0, S[row-1][col-1], S[row-1][col], S[row][col-1], S[row][col]])
                else:
                    ov = np.asarray([0, S[row-1][col-1], S[row][col-1], S[row-1][col], S[row][col]])
                S_c[row][col] = ov[1]  + ov[2]*0.75 + ov[3]*0.5 + ov[4]*0.25
            elif row > 0:
                S[row][col] = image[row][col] +  S[row - 1][col]
                ov = np.asarray([0, S[row - 1][col], S[row][col]])
                S_c[row][col] = ov[1]  + ov[2]*0.50
            elif col > 0:
                S[row][col] = image[row][col] +  S[row][col - 1]
                ov = np.asarray([0, S[row - 1][col], S[row][col]])
                S_c[row][col] = ov[1]  + ov[2]*0.50
            else:
                S[row][col] = image[row][col]
                S_c[row][col] = S[row][col]  
    return  S, S_c
``` 

**F(A4) Integral image: Choquet Integral Image**

```Python
def compute_summed_area_table_CHO(image ):
    height = len(image)
    width = len(image[0])
    S   = [[0.0] * width for _ in range(height)] 
    S_c = S# Create an empty summed area table
    for row in range(0, height):
        for col in range(0, width):
            if (row > 0) and (col > 0):
                S[row][col] = image[row][col] + S[row][col - 1] +  S[row - 1][col] -  S[row - 1][col - 1]
                if(S[row][col - 1] > S[row - 1][col] ):
                    ov = np.asarray([0, S[row-1][col-1], S[row-1][col], S[row][col-1], S[row][col]])
                else:
                    ov = np.asarray([0, S[row-1][col-1], S[row][col-1], S[row-1][col], S[row][col]])
                S_c[row][col] = (ov[1]-ov[0]) +(ov[2]-ov[1])*0.75 + (ov[3]-ov[2])*0.50 +(ov[4]-ov[3])*0.25
            elif row > 0:
                S[row][col] = image[row][col] +  S[row - 1][col]
                ov = np.asarray([0, S[row - 1][col], S[row][col]])
                S_c[row][col] =  (ov[1]-ov[0]) +(ov[2]-ov[1])*0.5
            elif col > 0:
                S[row][col] = image[row][col] +  S[row][col - 1]
                ov = np.asarray([0, S[row - 1][col], S[row][col]])
                mu_q = np.asarray([1, 0.5])
                S_c[row][col] =  (ov[1]-ov[0]) +(ov[2]-ov[1])*0.6
            else:
                S[row][col] = image[row][col]
                S_c[row][col] = S[row][col]  
    return  S, S_c
``` 

**F(A3) Integral image: Hamacher t-norm Integral Image**

```Python
def compute_summed_area_table_HAM(image ):
    height = len(image)
    width = len(image[0])
    S   = [[0.0] * width for _ in range(height)]
    S_c = S# Create an empty summed area table
    for row in range(0, height):
        for col in range(0, width):
            if (row > 0) and (col > 0):
                S[row][col] = image[row][col] + S[row][col - 1] +  S[row - 1][col] -  S[row - 1][col - 1]
                if(S[row][col - 1] > S[row - 1][col] ):
                    ov = np.asarray([0, S[row-1][col-1], S[row-1][col], S[row][col-1], S[row][col]])
                else:
                    ov = np.asarray([0, S[row-1][col-1], S[row][col-1], S[row-1][col], S[row][col]])
                S_c[row][col] = (ov[1]-ov[0])/(ov[1] + 1 - (ov[1]))+(ov[2]-ov[1])/(ov[2] + 0.75 - (ov[2]*0.75)) + (ov[3]-ov[2])//(ov[3] + 0.50 - (ov[0]*0.50)) +(ov[4]-ov[3])/(ov[4] + 0.25 - (ov[4]*0.25))
            elif row > 0:
                S[row][col] = image[row][col] +  S[row - 1][col]
                ov = np.asarray([0, S[row - 1][col], S[row][col]])
                S_c[row][col] =  (ov[1]-ov[0])/(ov[1] + 1 - (ov[1]))+(ov[2]-ov[1])/(ov[2] + 0.5 - (ov[2]*0.5))
            elif col > 0:
                S[row][col] = image[row][col] +  S[row][col - 1]
                ov = np.asarray([0, S[row - 1][col], S[row][col]])
                S_c[row][col] =  (ov[1]-ov[0])/(ov[1] + 1 - (ov[1]))+(ov[2]-ov[1])/(ov[2] + 0.5 - (ov[2]*0.5))
            else:
                S[row][col] = image[row][col]
                S_c[row][col] = S[row][col]  
    return  S, S_c


```




**Fuzzy adaptive thresholding of Images in [0,1] with with one of the above Fuzzy integral images**

```Python
def adaptive_thresh_fuzzy_int(input_img, int_img, a1=4, a2=1, T=0, log=False):    
    out_img = np.zeros_like(input_img) 
    mat     = out_img
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
            if count == 0:
                if x0 == x1 and y0 == y1:
                    sum_ = int_img[y0, x0]
                if x1 == x0 and y0 != y1:
                    sum_ = int_img[y1, x1] - int_img[y0, x1]
                if y1 == y0 and x1 != x0:
                    sum_ = int_img[y1, x1] - int_img[y1, x0]
            else:
                sum_ = int_img[y1, x1] - int_img[y0, x1] - int_img[y1, x0] + int_img[y0, x0]           
            
            mat[row,col] = sum_/count

            if input_img[row, col]*count  < sum_ * (1.-T).:
                out_img[row,col] = 0
            else:
                out_img[row,col] = 1

    return out_img, mat, T
```






## Toy dataset, Test datased and an additional theta-dataset for thresholding
The respective datasets are: [gamma-dataset](gamma-dataset),  [theta-dataset](theta-dataset) and .
The gamma-dataset is our challenging dataset, while the theta-dataset is provided by an external source, please refeer to the citations in the paper.  

 

**Licence**
The same of IEEE transactions for Journal Papers.

This work is supported by the Artificial Intelligence departement of the University of Navarra - UPNA (SP) and by the DISA-MIS department, NeuRoNe Lab (University of Salerno - IT).



** .BIB **

