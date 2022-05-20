## Sugeno integral generalization applied to improve adaptive image binarization
This repository contains the manuscript mentioned in the title in Latex/Overleaf format [here](https://www.overleaf.com/read/ykcjbqjvhrmt), and associated code and data sets used for testing our novel methodology. Should you need help running our code, please contact us.


### Abstract
Classic adaptive binarization methodologies threshold pixels intensity with re-spect  to  adjacent  pixels  exploiting  integral  images.   In  turn,  integral  imagesare  generally  computed  optimally  by  using  the  summed-area-table  algorithm(SAT).  This  document  presents  a  new  adaptive  binarization  technique  basedon  fuzzy  integral  images.   Which,  in  turn,  this  technique  is  supported  by  anefficient design of a modified SAT for generalized Sugeno fuzzy integrals.  Wedefine this methodology as FLAT (Fuzzy Local Adaptive Thresholding).  Exper-imental results show that the proposed methodology produced a better imagequality thresholding than well-known global and local thresholding algorithms.We proposed new generalizations of different fuzzy integrals to improve existingresults and reaching an accuracy≈0.94 on a wide dataset.  Moreover, due tohigh performances, these new generalized Sugeno fuzzy integrals created ad hocfor adaptive binarization, can be used as tools for grayscale processing and morecomplex real-time thresholding applications.

 

---

**Journal Citation**

Bardozzo, Francesco, et al. "Sugeno integral generalization applied to improve adaptive image binarization." Information Fusion - Elsevier (2020).

[read here](https://www.sciencedirect.com/science/article/pii/S1566253520304012)

```
@article{bardozzo2020sugeno,
  title={Sugeno integral generalization applied to improve adaptive image binarization},
  author={Bardozzo, Francesco and De La Osa, Borja and Horansk{\'a}, L'ubom{\'\i}ra and Fumanal-Idocin, Javier and delli Priscoli, Mattia and Troiano, Luigi and Tagliaferri, Roberto and Fernandez, Javier and Bustince, Humberto},
  journal={Information Fusion},
  year={2020},
  publisher={Elsevier}
}

```
---

**Conference Citation**
 
Fumanal-Idocin, Javier, et al. "Gated Local Adaptive Binarization using Supervised Learning."  WILF 2021: International Workshop on Fuzzy Logic and Applications. (2021)

[read here](http://ceur-ws.org/Vol-3074/paper16.pdf)

```
@inproceedings{fumanal2021gated,
  title={Gated Local Adaptive Binarization using Supervised Learning},
  author={Fumanal-Idocin, Javier and Uriarte, Juan and de la Osa, Borja and Bardozzo, Francesco and Fern{\'a}ndez, Javier and Bustince, Humberto},
  booktitle={WILF 2021: International Workshop on Fuzzy Logic and Applications},
  year={2021},
  organization={WILF}
}
```


---

 
## Supplmentary Materials
**Table S-1** : 
In the [Table S-1](/adaptive_bin_supp.pdf) a comparison between our algorithms and the Bradley algorithm on the toy-dataset is provided. The values of perturbations of each Image are indicated with respective values, as it is described in the paper. 

**Sensitivity and Robustness analysis table for FLAT**: are provided in this [file](sensitivity_robustness)

**Time benchmark on Google Colab**: The JIT optimized implementations of the CF12 - FLAT (the best performing one in terms of SSIM, MSE, Precision and Recall) and the associated benchmarks are analyzed on Google Colab and they are provided in the following link: [link](https://colab.research.google.com/drive/1bdL0tHnW213_AZUHoYX2vq1PhEC75kCA
).

 
 
## FLAT algortihm (Fuzzy Local Adaptive Thresholding)

Here a whole overview of the FLAT algorithm. More details in the paper. 

![alt text](/image1git.png)


## Toy dataset, Test dataset and an additional theta-dataset for thresholding
The respective datasets are: [toy dataset](gamma-dataset),  [test dataset 280](https://drive.google.com/drive/folders/11lIv91rRgYFbADVPptOsLnEu2zJDCCnJ?usp=sharing) of 280 images for testing our algorithms at the optimum, and [test_dataset 2413](https://drive.google.com/open?id=15OezFUWvfXpYI3Tx8gqneC8_ETND8CdJ) for testing our and other thresholding algorithms with a bigger benchmark (2413 images).

The toy dataset is our challenging dataset with controlled perturbations, while the 2 test datasets are provided by an [external source](https://github.com/Andrew-Qibin/DSS), please refeer to citations in the paper for more details. 


**Visual Examples**

Here, an additional visual examples on another dataset with GT (we call [theta-dataset](/theta-dataset)) for our 3 methods **A2,A3,A4** with respect 4 traditional binarization methods (note Global Th is the Otsu method) is provided:

![alt text](/res0.png)


---

## Source code 

The extened tests and implementations are provided in a single Python 3.6.8 script, [here](/fuzzy_adaptive_bin.py). However, the core of the research is provided here as follows:

**Thresholding algorithms based on integral images with Sugeno generalizations and Bradley**

**(Python Class implementation optimized with Numba)**

```Python
import warnings
import sys
from numba import int32, float32
from numba.experimental import jitclass
import numpy as np


if not sys.warnoptions:
    warnings.simplefilter("ignore")

spec = [
    ('image', float32[:, :]),
    ('gt', float32[:, :]),
    ('height', int32),
    ('width', int32),
    ('S', float32[:, :]),
    ('S_c', float32[:, :]),
    ('out_img', float32[:, :]),
    ('th_mat', float32[:, :]),
]


@jitclass(spec)
class fuzzy_sat:
    def __init__(self, image):
        shape = image.shape
        self.image = np.asarray(image, dtype=float32)
        self.height = shape[0]
        self.width = shape[1]
        self.S = np.zeros(shape, dtype=float32)  # Create an empty summed area table
        self.S_c = np.zeros(shape, dtype=float32)  # Create an empty summed area table
        self.out_img = np.zeros(shape, dtype=float32)
        self.th_mat = np.zeros(shape, dtype=float32)

    def compute_sat(self):                              #Integral image
        for row in range(0, self.height):
            for col in range(0, self.width):
                if (row > 0) and (col > 0):
                    self.S[row][col] = self.image[row][col] + self.S[row][col - 1] + self.S[row - 1][col] - \
                                       self.S[row - 1][col - 1]
                elif row > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row - 1][col]
                elif col > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row][col - 1]
                else:
                    self.S[row][col] = self.image[row][col]

    def compute_sat_cf12(self):                              #CFval 1,2 integral image
        for row in range(0, self.height):
            for col in range(0, self.width):
                if (row > 0) and (col > 0):
                    self.S[row][col] = self.image[row][col] + self.S[row][col - 1] + self.S[row - 1][col] - \
                                       self.S[row - 1][col - 1]
                    if (self.S[row][col - 1] > self.S[row - 1][col]):
                        ov = np.asarray(
                            [0, self.S[row - 1][col - 1], self.S[row - 1][col], self.S[row][col - 1], self.S[row][col]])
                    else:
                        ov = np.asarray(
                            [0, self.S[row - 1][col - 1], self.S[row][col - 1], self.S[row - 1][col], self.S[row][col]])
                    self.S_c[row][col] = ov[1] + ov[2] * 0.75 + ov[3] * 0.5 + ov[4] * 0.25
                elif row > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row - 1][col]
                    ov = np.asarray([0, self.S[row - 1][col], self.S[row][col]])
                    self.S_c[row][col] = ov[1] + ov[2] * 0.50
                elif col > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row][col - 1]
                    ov = np.asarray([0, self.S[row - 1][col], self.S[row][col]])
                    self.S_c[row][col] = ov[1] + ov[2] * 0.50
                else:
                    self.S[row][col] = self.image[row][col]
                    self.S_c[row][col] = self.image[row][col]

    def compute_sat_cho(self):                            #Choquet integral image
        for row in range(0, self.height):
            for col in range(0, self.width):
                if (row > 0) and (col > 0):
                    self.S[row][col] = self.image[row][col] + self.S[row][col - 1] + self.S[row - 1][col] - \
                                       self.S[row - 1][col - 1]
                    if (self.S[row][col - 1] > self.S[row - 1][col]):
                        ov = np.asarray(
                            [0, self.S[row - 1][col - 1], self.S[row - 1][col], self.S[row][col - 1], self.S[row][col]])
                    else:
                        ov = np.asarray(
                            [0, self.S[row - 1][col - 1], self.S[row][col - 1], self.S[row - 1][col], self.S[row][col]])
                    self.S_c[row][col] = (ov[1] - ov[0]) + (ov[2] - ov[1]) * 0.75 + (ov[3] - ov[2]) * 0.50 + (
                            ov[4] - ov[3]) * 0.25
                elif row > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row - 1][col]
                    ov = np.asarray([0, self.S[row - 1][col], self.S[row][col]])
                    self.S_c[row][col] = (ov[1] - ov[0]) + (ov[2] - ov[1]) * 0.5
                elif col > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row][col - 1]
                    ov = np.asarray([0, self.S[row - 1][col], self.S[row][col]])
                    self.S_c[row][col] = (ov[1] - ov[0]) + (ov[2] - ov[1]) * 0.5
                else:
                    self.S[row][col] = self.image[row][col]
                    self.S_c[row][col] = self.S[row][col]

    def compute_sat_ham(self):                  #Hamacher integral image
        for row in range(0, self.height):
            for col in range(0, self.width):
                if (row > 0) and (col > 0):
                    self.S[row][col] = self.image[row][col] + self.S[row][col - 1] + self.S[row - 1][col] - \
                                       self.S[row - 1][col - 1]
                    if (self.S[row][col - 1] > self.S[row - 1][col]):
                        ov = np.asarray(
                            [0, self.S[row - 1][col - 1], self.S[row - 1][col], self.S[row][col - 1], self.S[row][col]])
                    else:
                        ov = np.asarray(
                            [0, self.S[row - 1][col - 1], self.S[row][col - 1], self.S[row - 1][col], self.S[row][col]])
                    self.S_c[row][col] = (ov[1] - ov[0]) / (ov[1] + 1 - (ov[1])) + (ov[2] - ov[1]) / (
                            ov[2] + 0.75 - (ov[2] * 0.75)) + (ov[3] - ov[2]) // (ov[3] + 0.50 - (ov[0] * 0.50)) + (
                                                 ov[4] - ov[3]) / (ov[4] + 0.25 - (ov[4] * 0.25))
                elif row > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row - 1][col]
                    ov = np.asarray([0, self.S[row - 1][col], self.S[row][col]])
                    self.S_c[row][col] = (ov[1] - ov[0]) / (ov[1] + 1 - (ov[1])) + (ov[2] - ov[1]) / (
                            ov[2] + 0.5 - (ov[2] * 0.5))
                elif col > 0:
                    self.S[row][col] = self.image[row][col] + self.S[row][col - 1]
                    ov = np.asarray([0, self.S[row - 1][col], self.S[row][col]])
                    self.S_c[row][col] = (ov[1] - ov[0]) / (ov[1] + 1 - (ov[1])) + (ov[2] - ov[1]) / (
                            ov[2] + 0.5 - (ov[2] * 0.5))
                else:
                    self.S[row][col] = self.image[row][col]
                    self.S_c[row][col] = self.S[row][col]

    def adaptive_thresh_bradley(self, a1, T):                    #adaptive thresholding
        w_n = min(self.height, self.width) / a1
        for col in range(self.width):
            for row in range(self.height):
                # SxS region
                y0 = int(max(row - w_n, 0))
                y1 = int(min(row + w_n, self.height - 1))
                x0 = int(max(col - w_n, 0))
                x1 = int(min(col + w_n, self.width - 1))

                count = (y1 - y0) * (x1 - x0)
                sum_ = self.S[y1, x1] - self.S[y0, x1] - self.S[y1, x0] + self.S[y0, x0]

                self.th_mat[row, col] = sum_ / count

                if self.image[row, col] * count < sum_ * (1. - T):
                    self.out_img[row, col] = 0
                else:
                    self.out_img[row, col] = 1

    def adaptive_thresh_fuzzy(self, a1, T):                     #fuzzy adaptive thresholding
        w_n = min(self.height, self.width) / a1
        for col in range(self.width):
            for row in range(self.height):
                # SxS region
                y0 = int(max(row - w_n, 0))
                y1 = int(min(row + w_n, self.height - 1))
                x0 = int(max(col - w_n, 0))
                x1 = int(min(col + w_n, self.width - 1))

                count = (y1 - y0) * (x1 - x0)
                sum_ = self.S_c[y1, x1] - self.S_c[y0, x1] - self.S_c[y1, x0] + self.S_c[y0, x0]

                self.th_mat[row, col] = sum_ / count

                if self.image[row, col] * count < sum_ * (1. - T):
                    self.out_img[row, col] = 0
                else:
                    self.out_img[row, col] = 1

    def get_S(self):
        return (self.S)

    def get_S_c(self):
        return (self.S_c)

    def get_FTh(self):
        return (self.out_img)



```

**Class usage:**

```
int_img = fuzzy_sat(np.asarray(test_image))
int_img.compute_sat_cf12()
out_img = int_img.get_FTh()

```


---





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


**Licence**
The same of Information Fusion Journal - Elsevier

This work is supported by the Artificial Intelligence departement of the University of Navarra - UPNA (SP) and by the DISA-MIS department, NeuRoNe Lab (University of Salerno - IT).


