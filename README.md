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
 The experimental results show that the proposed methodology have produced an image quality thresholding often better than other traditional or simple neural network models. Furthermore, this new methodology is also competitive in terms of performance over time. Therefore, these new generalized fuzzy integrals (based on generalized form of Sugeno, Choquet and Hamacher t-Norm) can be used as a tool for grayscale processing in real-time and deep-learning applications.			
 
 **Source Code**
 Source for the FLAT methos are in this repository.
 The E.coli [whole metabolic network](/ecocyc.kegg.igraph.Rdata) is integrated from [KEGG](http://www.genome.jp/kegg/) and [EcoCyc](https://ecocyc.org/).
 
### theta-dataset and gamma-dataset


### FLAT algortihm (Fuzzy Local Adaptive Thresholding)
Here a whole overview of the FLAT algorithm.  In Boxes a-b, the steps of FLAT for the computation of both the integral image S and the fuzzy integral image $F_{A_i}$ are shown.  The blue square in Box a represents the current value of p(x,y) in I, (in red) and its neighboring pixels defined in the $j$-th operative window. For each pixel $p(x,y)$ and for every  \emph{j}-th operative window (yellow box), the 4 values in $S$ are mapped with the associated fuzzy measures through $F_{A_i}(x,y)=f_i(\vec{ov}, \vec{m})$ for $i=1,2,3,4$. The output of the fuzzy-based integral functional computation is saved in the fuzzy integral image $F_{A_i}$. This computation is described in the formula \ref{fuz}. In the \emph{j}-th operative window, the fixed values of min $v_1$ and max $v_4$ are represented in violet and blue. For the decision of $v_2$ and $v_3$, the green arrow indicates the $P_{swap}$ action as described in the Procedure \ref{eq6}. In Figure \ref{fig_000} Box \textbf{(c)}, the \emph{k}-th local search window $w_n$ used for the locally adaptive thresholding is shown. It is important to underline that, as it is described in the Algorithm \ref{alg:algo2}, only the 4 values in the orange rectangles are used for the binarization. These 4 values are not necessarily adjacent like in the operative window. The dashed red arrows show the local window sliding directions, from up to down, from left to right. The local window has a fixed size of $n_a * n_a$. The $I_b$ indicates the binarized image given in output considering the $b$-type fuzzy integral image.

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

