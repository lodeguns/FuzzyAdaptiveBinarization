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
 
 **Source Code**
 Source for the FLAT methos are in this repository.
 The E.coli [whole metabolic network](/ecocyc.kegg.igraph.Rdata) is integrated from [KEGG](http://www.genome.jp/kegg/) and [EcoCyc](https://ecocyc.org/).
 
### theta-dataset and gamma-dataset


### FLAT algortihm (Fuzzy Local Adaptive Thresholding)
Here a whole overview of the FLAT algorithm. More details in the paper. 
![alt text](/image1git.png)




Here, for the impatient, is an implementation of the FLAT methods in [Python](https://cran.r-project.org/)

**Fuzzy summed-area table algorithm**

**Classic summed-area table algorithm (SAT)**

```Python

def compute_summed_area_table(image):
    # image is a 2-dimensional array containing ints or floats, with at least 1 element.
    height = len(image)
    width = len(image[0])
    new_image = [[0.0] * width for _ in range(height)] # Create an empty summed area table
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

**CF12 - Generalized Sugeno (A2)**
```Python
def compute_summed_area_table_F1F2(image ):
    # image is a 2-dimensional array containing ints or floats, with at least 1 element.
    height = len(image)
    width = len(image[0])
    S   = [[0.0] * width for _ in range(height)] # Create an empty summed area table
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
**Choquet Integral Image (A4)**
```Python
def compute_summed_area_table_CHO(image ):
    # image is a 2-dimensional array containing ints or floats, with at least 1 element.
    height = len(image)
    width = len(image[0])
    S   = [[0.0] * width for _ in range(height)] # Create an empty summed area table
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

**Hamacher t-norm Integral Image (A3)**
```Python
def compute_summed_area_table_HAM(image ):
    # image is a 2-dimensional array containing ints or floats, with at least 1 element.
    height = len(image)
    width = len(image[0])
    S   = [[0.0] * width for _ in range(height)] # Create an empty summed area table
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




**Fuzzy adaptive thresholding with Generalized Sugeno (A3)**

**Fuzzy adaptive thresholding with Generalized Sugeno (A4)**

**Fuzzy adaptive thresholding with Generalized Sugeno (A1)**

** Google colab optimized implementations **
In the following links the optimized implementations of the FLAT algorithms and the associated benchmarks with Google Colab . 





**Licence**
BMC Bioinformatics, This work is supported by NeuRoNe Lab (University of Salerno - IT) and the Artificial Intelligence departement of the University of Navarra - UPNA (SP).



** .BIB **

