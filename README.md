## Adaptive binarization based on fuzzy integrals
This repository contains the manuscript mentioned in the title, and associated code and data sets used for testing our novel methodology. Should you need help running our code, please contact us.

### Preprint Citation

Bardozzo, Francesco, et al. "Adaptive binarization based on fuzzy integrals." arXiv preprint arXiv:2003.08755 (2020).

@article{bardozzo2020adaptive,
  title={Adaptive binarization based on fuzzy integrals},
  author={Bardozzo, Francesco and De La Osa, Borja and Horanska, Lubomira and Fumanal-Idocin, Javier and Troiano, Luigi and Tagliaferri, Roberto and Fernandez, Javier and Bustince, Humberto and others},
  journal={arXiv preprint arXiv:2003.08755},
  year={2020}
}

### Abstract
Adaptive binarization methodologies threshold the intensity of the pixels with respect to adjacent pixels exploiting the integral images. In turn, the integral images are generally computed optimally by using the summed-area-table algorithm (SAT). This document presents a new adaptive binarization technique based on fuzzy integral images through an efficient design of a modified SAT for fuzzy integrals. We define this new methodology as FLAT (Fuzzy Local Adaptive Thresholding). Experimental results show that the proposed methodology produced a better image quality thresholding than well-known global and local algorithms based on thresholds.  We proposed new generalizations of different fuzzy integrals to improve existing results and reaching an accuracy 0.94 on an an extended dataset. Moreover,  due to the high performances, these new generalized fuzzy integrals created ah hoc for adaptive binarization, can be used as tools for grayscale processing and more complex real-time applications. 	
 
## Supplmentary Materials
**Table S-1** : 
In the [Table S-1](/adaptive_bin_supp.pdf) a comparison between our algorithms and the Bradley algorithm on the toy-dataset is provided. The values of perturbations of each Image are indicated with respective values, as it is described in the paper. 

**Sensitivity and Robustness analysis table for FLAT**: are provided in this [file](sensitivity_robustness)

**Time benchmark on Google Colab**: The JIT optimized implementations of the CF12 - FLAT (the best performing one in terms of SSIM, MSE, Precision and Recall) and the associated benchmarks are analyzed on Google Colab and they are provided in the following link: [link](https://colab.research.google.com/drive/1bdL0tHnW213_AZUHoYX2vq1PhEC75kCA
).

 
 
## FLAT algortihm (Fuzzy Local Adaptive Thresholding)

Here a whole overview of the FLAT algorithm. More details in the paper. 

![alt text](/image1git.png)


**Source code**: 
The extened tests and implementations are provided in a single Python 3.6.8 script, [here](/fuzzy_adaptive_bin.py).


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






## Toy dataset, Test dataset and an additional theta-dataset for thresholding
The respective datasets are: [toy dataset](gamma-dataset),  [test dataset 280](https://drive.google.com/drive/folders/11lIv91rRgYFbADVPptOsLnEu2zJDCCnJ?usp=sharing) of 280 images for testing our algorithms at the optimum, and [test_dataset 2413](https://drive.google.com/open?id=15OezFUWvfXpYI3Tx8gqneC8_ETND8CdJ) for testing our and other thresholding algorithms with a bigger benchmark (2413 images).

The toy dataset is our challenging dataset with controlled perturbations, while the 2 test datasets are provided by an [external source](https://github.com/Andrew-Qibin/DSS), please refeer to citations in the paper for more details. 


**Visual Examples**

Here, an additional visual examples on another dataset with GT (we call [theta-dataset](/theta-dataset)) for our 3 methods **A2,A3,A4** with respect 4 traditional binarization methods (note Global Th is the Otsu method) is provided:

![alt text](/res0.png)

 

**Licence**
The same of Information Fusion Journal

This work is supported by the Artificial Intelligence departement of the University of Navarra - UPNA (SP) and by the DISA-MIS department, NeuRoNe Lab (University of Salerno - IT).



** .BIB ** as soon as possible.

