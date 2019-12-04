## Adaptive binarization based on fuzzy integrals

This repository contains the manuscript mentioned in the title, and associated code and data sets used for testing our novel methodology. Should you need help running our code, please contact us.

https://link

### Citation

Bardozzo et al.  "Adaptive binarization based on fuzzy integrals" Journal Journal.

### Abstract
**Background:**  	 
Adaptive binarization methodologies thredhold the intensity of the pixels with respect to adjacent pixels exploiting the integral images. In turn, the integral images are generally computed optimally using the summed-area-table algorithm (SAT).                                 
**Results:**   
This document presents a new adaptive binarization technique based on fuzzy integral images through an efficient design of a modified SAT for fuzzy integrals. We define this new methodology as FLAT (Fuzzy Local Adaptive Thresholding). Furthermore, a new definition of generalized high performance fuzzy integral is provided. 		
                                
 **Conclusions:** 
 The experimental results show that the proposed methodology have produced an image quality thresholding often better than other traditional or simple neural network models. We propose a new generalization of the Sugeno and CF12 integrals to improve the existing results and how they can be efficiently computed in the Integral Image. Therefore, these new generalized fuzzy integrals can be used as a tool for grayscale processing in real-time and deep-learning applications.			
 
 **Source Code**
 Source for the FLAT methos are in this repository.
 The E.coli [whole metabolic network](/ecocyc.kegg.igraph.Rdata) is integrated from [KEGG](http://www.genome.jp/kegg/) and [EcoCyc](https://ecocyc.org/).
 
### theta-dataset and gamma-dataset


### FLAT algortihm (Fuzzy Local Adaptive Thresholding)
Here a whole overview of the FLAT algorithm. More details in the paper. 
![alt text](/image1git.png)




Here, for the impatient, is an implementation of the FLAT methods in [Python](https://cran.r-project.org/)

**Fuzzy summed-area table algorithm**
```Python

}
```



**Fuzzy adaptive thresholding with Generalized Sugeno (A2)**

**Fuzzy adaptive thresholding with Generalized Sugeno (A3)**

**Fuzzy adaptive thresholding with Generalized Sugeno (A4)**

**Fuzzy adaptive thresholding with Generalized Sugeno (A1)**

** Google colab optimized implementations **
In the following links the optimized implementations of the FLAT algorithms and the associated benchmarks with Google Colab . 





**Licence**
BMC Bioinformatics, This work is supported by NeuRoNe Lab (University of Salerno - IT) and the Artificial Intelligence departement of the University of Navarra - UPNA (SP).



** .BIB **

