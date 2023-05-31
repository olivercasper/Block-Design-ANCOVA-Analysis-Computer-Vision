# Randomized Block Design: Quantifying Treatment Effect with Computer Vision 

Experimental results from image data using R and OpenCV2 for Python 


Contents
1.	Introduction
2.	Methods
2.1	  Experimental Design
2.1.1	Sample
2.1.2	Procedure
2.2	Computer Vision
2.3	Statistical Methods
2.3.1	Measures 
2.3.2	Analysis
3.	Results
4.	Discussion
5.	Appendix
5.1	   Data
5.2	   R Code
5.3	   Python Code
6.	References
 


QAC 307

Impaired Proliferation in Bacterial Cultures Treated with UV Irradiation


Oliver Biele
12-9-2022

 
Contents
	Introduction
	Methods
	  Experimental Design
	Sample
	Procedure
	Computer Vision
	Statistical Methods
	Measures 
	Analysis
	Results
	Discussion
	Appendix
	   Data
	   R Code
	   Python Code
	References
 
1. Introduction
	Bacteria are easily the most abundant organisms on Earth, representing one of the three basic domains of life. These single-celled prokaryotes are found nearly everywhere and serve a diverse set of biological roles, ranging from symbiotic to pathogenic. As such, bacterial contamination poses a threat to public health which has been addressed in supply chains. One interesting approach to sterilization is the application of ultraviolet (UV) irradiation.
 Exposure to ultraviolet light damages the structure of DNA molecules, preventing mitotic replication of bacterial cells. Researchers have demonstrated the efficacy of UV irradiation in the sterilization of both dairy products (Pereira, Bicalho et al. 2014) as well as drinking water (Pullerits, Ahlinder et al. 2020). Our study aimed to assess the relationship between UV irradiance and relative extent of bacterial growth in agar medium. We presently discuss the experimental and statistical design we employed to explore the question: “How does the extent of bacterial growth over an incubation period vary with level of initial ultraviolet irradiation?” 

2. Methods

2.1 Experimental Design
2.1.1 Sample
	Bacteria were grown in a series of petri dishes, with each plate representing one observational unit. A nutritionally rich agar medium made with Luria broth was chosen here as the growth substrate. Initial bacterial inocula were collected from saliva using sterile cotton swabs and subsequently transferred to agar. Treatment was evenly and randomly assigned to units throughout the inoculation process through a randomized block design with order as the blocking variable. There were 6 levels of order representing inoculation from start to finish (3 units per level). In total, 18 inoculated plates were used in our sample, with 6 plates representing each level of UV treatment (N = 18; n = 6). 


2.1.2 Procedure
	To investigate the relative effect of UV irradiation on bacterial growth, plates were exposed to 3 different intensities of UV light for 45 seconds prior to a 3-day incubation period. The levels of treatment were either UV high, UV low, or no exposure (control). Plates assigned to UV high were irradiated at half the vertical distance (11 cm vs. 22 cm) . The UV low group therefore received a dose that was ¼ of the intensity of UV high due to the inverse square law. Plates of both treatments were arranged in 3x2 arrays on separate shelves and the no-light control plates were placed in a drawer. A single UV lamp, set to “high”, with two adjustable arms was used as the light source for both intensities. Following incubation, plates were photographed at the same distance and disposed of. The value of total bacterial growth was then calculated for each observation using a computer vision model. 

2.2 Computer Vision
	To approach the problem of accurately quantifying the growth response, we relied on various computer vision techniques to process and analyze the images of our plates post-incubation. The aim here was to count the number of pixels representing bacterial colonies and record that value as the growth response while ignoring all other pixels. We relied on the OpenCV library for Python 3.10.7 along with NumPy and Pillow libraries for the entire computer vision procedure. 
Our methods here involved two main components: thresholding and edge detection. Binary thresholding (Figure 1) was used to represent each image as a 2-dimensional binary matrix where the sum of elements yields the value of growth. Adaptive gaussian thresholding was used separately to prepare images for edge detection by maximizing the contrast between boundaries of discrete objects within the image (Figure 2). Contours were fit to the gaussian image and filtered for shape and area to approximate an ellipsoid mask which was used to remove non-zero pixels beyond the agar boarder in the corresponding binary image. The final result was a set of binary plate images containing only non-zero bacteria pixels (Figure 3). These results were then passed to OpenCV’s countNonZero() function to obtain the integer value of growth for each observation.   
Figure 1: Binary thresholding filter















Figure 2: Binary and adaptive Gaussian thresholding applied to greyscale image






Figure 3: Unwanted pixels are removed with masking image in shape of perimeter contour. Here the contours of the agar perimeter are fit in the Gaussian image with OpenCV’s canny edge detection to create an ellipsoid mask. The adaptive Gaussian filter provides the boundary contrast necessary for the machine learning model to accurately determine the contours and hierarchy of objects within the image. 











2.3 Statistical Methods
2.3.1 Measures
	Data were recorded for the assignment of treatment, order block, array position, and growth. The natural log of growth was used as the response variable in analysis. Treatment was coded as either “uv_high”, “uv_low”, or “control”. Position in inoculation order was coded with integers 1:6 as an ordinal variable. Row and column values for array position were recorded for UV treatments and coded with integers 1:3 and 1:2 respectively, with “3” being the bottom-most row and “2” being the right column. Control plates were assigned a value of “0” for row and column positions. The growth response was coded as a discrete quantitative variable with each unit increase representing a single non-zero pixel of the final result image. 
2.3.2 Analysis
	We used analysis of variance with the inclusion of covariates (ANCOVA) to determine how the extent of bacterial growth varies among the three levels of UV treatment. Our model is represented by the equation:

\mathbit{y}_{\mathbit{i},\mathbit{j},\mathbit{k},\mathbit{l}}=\mathbit{\mu}+\mathbit{\tau}_\mathbit{i}+\mathbit{\delta}_\mathbit{j}+\mathbit{\gamma}_\mathbit{k}+\mathbit{\varphi}_\mathbit{l}+\mathbit{\epsilon}_{\mathbit{i},\mathbit{j},\mathbit{k},\mathbit{l}}
H_0:\;\mu_{high}=\mu_{low}=\mu_c
H_A:\;\mathrm{At\ least\ one\ varies}
\alpha\ =\ 0.05

where τ, δ, γ, and φ are estimated effect terms of treatment, order, row position, column position respectively. Tukey’s multiple comparisons post-hoc was used to find the significant pairwise differences in effect across the 3 levels of treatment. All analysis was performed in R. The ggplot2 and gmodels R packages were used. 








3. Results  
	In line with our expectations, analysis of variance revealed overall difference in mean growth among UV treatments (F = 11.8 ; p = 0.0057) (figure 4). At α = 0.05, we reject the null hypothesis in favor of the alternate that at least one treatment mean varies. Tukey’s multiple comparisons post-hoc revealed that uv_high had a significantly lower value of log growth than both control (µhigh – µc = -0.71; p = 0.006) and uv_low (µhigh – µlow = -0.55; p = 0.02). There was no significant difference in log growth between uv_low and control (µlow  – µc = -0.15; p = 0.6). The blocking variable for inoculation order was also found to be significant (F = 11.4; p = 0.003) with growth displaying a positive trend with place in inoculation order (figure 5). Lastly, growth varied significantly among row positions (F = 6.2; p = 0.028) (figure 6). 

Figure 4: ln(growth) by treatment

Figure 5: ln(growth) by order block 

Figure 6: ln(growth) by row position
4. Discussion
	Based on the significant variance in mean growth between UV treatment levels (F = 11.8 ; p = 0.0057), we conclude that our experiment was mostly effective at meeting our endpoint. While the results do demonstrate the relationship of interest, the other significant factors in our model reveal flaws in procedural design and execution. The uneven growth across inoculation order indicates that transfer of bacterial material to the agar was not properly controlled. A future attempt of this experiment might consider using a volumetric inoculant instead of swabs. Furthermore, the significantly greater growth of the 3rd row suggests unequal irradiance of plates under the UV source. A square-shaped position array under a UV point source could possibly control relative irradiance here. Additionally, our statistical approach could be revised to balance position of replicates with row as a second block in a Latin square design. 
	Overall, we are pleased with the outcome of our experiment. In the present study, we set out to design experimental methods which would illustrate the impact of UV exposure on bacterial proliferation. We were interested to find not only this relationship, but also additional considerations for experimental design. The results of this analysis were a valuable lesson in unexpected outcomes and how data can reveal underlying experimental errors. 
 
5. Appendix
5.1 Data

 
 
 
 
5.2 R Code

require(ggplot2)
## Loading required package: ggplot2
require(gmodels)
## Loading required package: gmodels
## Inoculation Order
treatments <- c("UV_high","UV_low","Control")

df_order <- data.frame(plate=c(1:18),
                       treatment=c(replicate(6, sample(
                         treatments,3,replace=FALSE)))
                            )
df_order
##    plate treatment
## 1      1   UV_high
## 2      2   Control
## 3      3    UV_low
## 4      4   Control
## 5      5    UV_low
## 6      6   UV_high
## 7      7   Control
## 8      8   UV_high
## 9      9    UV_low
## 10    10   Control
## 11    11   UV_high
## 12    12    UV_low
## 13    13   UV_high
## 14    14   Control
## 15    15    UV_low
## 16    16   UV_high
## 17    17    UV_low
## 18    18   Control
## Treatment counts
table(df_order$treatment)
## 
## Control UV_high  UV_low 
##       6       6       6
## Inoculation order
list(df_order$treatment)
## [[1]]
##  [1] "UV_high" "Control" "UV_low"  "Control" "UV_low"  "UV_high" "Control"
##  [8] "UV_high" "UV_low"  "Control" "UV_high" "UV_low"  "UV_high" "Control"
## [15] "UV_low"  "UV_high" "UV_low"  "Control"
“UV_high” “UV_low” “Control” “Control” “UV_high” “UV_low” “UV_low” “UV_high” “Control” “UV_low” “UV_high” “Control” “Control” “UV_low” “UV_high” “UV_high” “UV_low” “Control”
#Data
df <- data.frame(plate=c(1:18),
                 treatment=c("UV_high", "UV_low",  "Control", "Control", "UV_high", "UV_low",  
                             "UV_low",  "UV_high", "Control", "UV_low", "UV_high", "Control", 
                             "Control", "UV_low",  "UV_high", "UV_high", "UV_low",  "Control"),
                 block=rep(c(1:6), each=3), 
                 row_pos=c(2,1,0,0,2,1,2,1,0,2,1,0,0,3,3,3,3,0), 
                 col_pos=c(1,1,0,0,2,2,1,1,0,2,2,0,0,1,1,2,2,0),
                 area=c(5072,13098,20247,16500,6695,16438,
                        23926,11473,35428,17890,10758,33088,
                        25549,20358, 21955,48781,66536,30833)
                 )
df$log_area <- log(df$area)

#Light sources placed 22 and 11 cm
D_h = "22cm"
D_l = "11cm"

#UV light applied to treatment plates for 45 seconds  
##Analysis
tapply(log(df$area), df$treatment, mean)
##   Control   UV_high    UV_low 
## 10.165626  9.460602 10.014835
#ANOVA
mod <- aov(log(area)~treatment + factor(block) + factor(row_pos) + factor(col_pos), data=df)
summary(mod)
##                 Df Sum Sq Mean Sq F value  Pr(>F)   
## treatment        2  1.654  0.8270  11.810 0.00571 **
## factor(block)    5  3.983  0.7966  11.376 0.00296 **
## factor(row_pos)  2  0.865  0.4326   6.178 0.02844 * 
## factor(col_pos)  1  0.147  0.1472   2.103 0.19033   
## Residuals        7  0.490  0.0700                   
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
TukeyHSD(mod)
##   Tukey multiple comparisons of means
##     95% family-wise confidence level
## 
## Fit: aov(formula = log(area) ~ treatment + factor(block) + factor(row_pos) + factor(col_pos), data = df)
## 
## $treatment
##                       diff        lwr        upr     p adj
## UV_high-Control -0.7050243 -1.1549670 -0.2550816 0.0059980
## UV_low-Control  -0.1507906 -0.6007333  0.2991520 0.6073818
## UV_low-UV_high   0.5542337  0.1042910  1.0041763 0.0201642
## 
## $`factor(block)`
##            diff         lwr       upr     p adj
## 2-1  0.10003856 -0.71871535 0.9187925 0.9961486
## 3-1  0.65942108 -0.15933283 1.4781750 0.1230972
## 4-1  0.51828695 -0.30046697 1.3370409 0.2715687
## 5-1  0.71295521 -0.10579870 1.5317091 0.0907318
## 6-1  1.43648940  0.61773548 2.2552433 0.0024665
## 3-2  0.55938252 -0.25937139 1.3781364 0.2166804
## 4-2  0.41824838 -0.40050553 1.2370023 0.4535786
## 5-2  0.61291665 -0.20583726 1.4316706 0.1603717
## 6-2  1.33645083  0.51769692 2.1552047 0.0037848
## 4-3 -0.14113414 -0.95988805 0.6776198 0.9821047
## 5-3  0.05353413 -0.76521978 0.8722880 0.9998056
## 6-3  0.77706831 -0.04168560 1.5958222 0.0631229
## 5-4  0.19466827 -0.62408565 1.0134222 0.9344796
## 6-4  0.91820245  0.09944854 1.7369564 0.0290417
## 6-5  0.72353419 -0.09521973 1.5422881 0.0854361
## 
## $`factor(row_pos)`
##            diff        lwr       upr     p adj
## 1-0 -0.03127594 -0.5966911 0.5341392 0.9976114
## 2-0 -0.18212533 -0.7475405 0.3832899 0.7191420
## 3-0  0.21340128 -0.3520139 0.7788165 0.6186414
## 2-1 -0.15084939 -0.7702307 0.4685319 0.8497133
## 3-1  0.24467722 -0.3747041 0.8640585 0.5867193
## 3-2  0.39552661 -0.2238547 1.0149079 0.2371376
## 
## $`factor(col_pos)`
##            diff        lwr       upr     p adj
## 1-0 -0.06395244 -0.5138951 0.3859902 0.9091437
## 2-0  0.06395244 -0.3859902 0.5138951 0.9091437
## 2-1  0.12790488 -0.3220378 0.5778475 0.6935385
## Plots
plot(mod)
    
ggplot(data=df)+
  geom_boxplot(aes(x=factor(treatment), y=log(area))) +
  ggtitle("Growth by Treatment") +
  xlab("Treatment")
 
ggplot(data=df)+
    stat_summary(aes(x=treatment, y=log(area)), fun=mean, geom="bar") +
    ggtitle("Growth by Treatment") +
    xlab("Treatment")
 
ggplot(data=df)+
  geom_boxplot(aes(x=factor(block), y=log(area))) +
  ggtitle("Growth by Inoculation Block") +
  xlab("Inoculation Group")
 
ggplot(data=df)+
  geom_boxplot(aes(x=factor(row_pos, labels=c("None (Control)", "Top", "Middle", "Lower")), y=log(area))) +
  ggtitle("Growth by Row Positition") +
  xlab("Row")
 
##Sample size calculations

#All treatments vary

#Growth
gm <- mean(df$log_area)

tau_uv <- tapply(df$log_area, INDEX=df$treatment, FUN=mean) - gm
ms_res = 0.07
diff <- min(abs(apply(combn(c(tau_uv),2), 2, diff)))
delta <- (diff)/sqrt(ms_res)

n <- ceiling(2*(qnorm(1-0.025) + qnorm(0.8))^2/(delta)^2)

N <- n*3

print(n)
## [1] 49
print(N)
## [1] 147



















5.2 Python Code

############################
## QAC 307 Final Project
############################
import os
import math
import numpy as np
import pandas as pd
import cv2 as cv
import imutils

from matplotlib import pyplot as plt
from scipy.interpolate import splprep, splev
from PIL import Image 
from pathlib import Path

import jupyterthemes as jt
from jupyterthemes import get_themes
from jupyterthemes.stylefx import set_nb_theme
from IPython.display import display, HTML

display(HTML(data="""
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 65%; }
    div#maintoolbar-container { width: 99%; }
</style>
"""))
set_nb_theme('oceans16')
sample_path = r"C:\Users\olive\Documents\QAC 307\images\sample.jpg"

img = cv.imread(sample_path,cv.COLOR_RGB2BGR)
gs = cv.imread(sample_path, cv.IMREAD_GRAYSCALE)

Image.fromarray(gs)
gs
array([[46, 46, 46, ..., 59, 63, 71],
       [46, 46, 46, ..., 73, 54, 64],
       [46, 46, 46, ..., 70, 63, 68],
       ...,
       [46, 46, 46, ..., 51, 57, 68],
       [46, 46, 46, ..., 51, 65, 71],
       [46, 46, 46, ..., 55, 67, 70]], dtype=uint8)
## Convert greyscale image to binary 

## Thresholding 
gs = cv.imread(sample_path, cv.IMREAD_GRAYSCALE)
gs_b = cv.medianBlur(gs,5)

ret,thresh1 = cv.threshold(gs_b,100,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(gs_b,120,255,cv.THRESH_BINARY)
ret,thresh3 = cv.threshold(gs_b,150,255,cv.THRESH_BINARY)
ret,thresh4 = cv.threshold(gs_b,190,255,cv.THRESH_BINARY)
ret,thresh5 = cv.threshold(gs_b,210,255,cv.THRESH_BINARY)

# plot
cap = '*In binary thresholding, contrast between objects and the background is maximized. '\
'\n' + r'The $\alpha$ channel of each pixel in the greyscale image is compared against a '\
'threshold \nvalue and set to either 1 or 0 in the output result.'
titles = ['Greyscale','t = 100','t = 120','t = 150','t = 190','t = 210']
images = [gs, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.suptitle('Binary image conversion with threshold t = [0 ... 255]', size=12, y=1.0)
plt.text(-570, 350, cap)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()

## Gaussian thresholding for dish perimeter
blur = cv.GaussianBlur(gs, (9,9), 0)

ret,th1 = cv.threshold(gs,190,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

# plot
titles = ['Greyscale', 'Binary','Gaussian']
images = [gs, th1, th2]
for i in range(3):
    plt.subplot(1,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.suptitle('Adaptive Gaussian Thresholding', size=12, y=0.8)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.5)
plt.show()
## Contours

## Scale contour function
def scale_contour(cnt, scale):
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled
#create an empty images for contours
gs_copy = gs.copy()
img_copy = img.copy()
img_contours = np.zeros(img.shape)
ext_contours = np.zeros(img.shape)
comb_contours = np.zeros(img.shape)
masking_img = gs.copy()
scale_ex = np.zeros(img.shape)

## Masking
stencil = np.zeros(gs_copy.shape).astype(gs_copy.dtype)
stencil_1 = np.zeros(gs_copy.shape).astype(gs_copy.dtype)
color = [255, 255, 255]

##Internal contours
contours, hierarchy = cv.findContours(th1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
int_conts = contours

# draw the internal contours on the empty image
cv.drawContours(img_contours, contours, -1, (0,255,0), 1)
cv.drawContours(img_copy, contours, -1, (0,255,0), 1)
cv.drawContours(scale_ex, contours, -1, (0,255,0), 1)
final_conts_img = img_contours.copy()

##Perimeter contour
thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,23,3)

cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# draw perimeter contour
lst = []
for c in cnts: # Get contour with greatest area
    area = cv.contourArea(c)
    lst.append(area)
ext_bound = max(lst)

## Scale perimeter contour by 0.98 to remove spurious non-zero pixels and draw on images
cont_ext = -1
for c in cnts:
    area = cv.contourArea(c)
    if area == ext_bound:
        cont_ext = c
        cv.drawContours(final_conts_img, [scale_contour(c, 0.98)], -1, (255,0,0), 1)
        cv.drawContours(ext_contours, [scale_contour(c, 0.98)], -1, (255,0,0), 1)
        cv.fillPoly(stencil, [scale_contour(c, 0.98)], color)
        cv.fillPoly(stencil_1, [c], color)
        cv.drawContours(masking_img, [scale_contour(c, 0.98)], -1, (255,0,0), 1)
        cv.drawContours(scale_ex, [scale_contour(c, 0.98)], -1, (255, 191, 0), 1)
        cv.drawContours(scale_ex, [c], -1, (255,0,0), 1)
        
## Masking        
result = cv.bitwise_and(gs_copy, stencil)
result_bad = cv.bitwise_and(gs_copy, stencil_1)
final = result.copy()
cv.drawContours(final, contours, -1, (0,255,0), 1)


# plot 
titles = ['Greyscale', 'Internal', 'Agar Perimeter', 'Contours']
images = [gs, img_contours, ext_contours, final_conts_img]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([]) 
plt.suptitle('Find contours of image', size=12, y=1)
plt.text(-130, -200, r'Binary $\longrightarrow$')
plt.text(-330, -65, r'Gaussian $\downarrow$')
plt.text(-420, 300, "*The pixels of interest are contained within the red perimeter contour. \n"\
        "This boundary will be used to remove all other pixels from the image.")
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.45)
plt.show()
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



## Remove pixels beyond perimeter

ret,thresh_f = cv.threshold(result,190,255,cv.THRESH_BINARY)

# plot 
titles = ['Greyscale','Mask Applied', 'Target Area']
images = [gs, result, thresh_f]

for i in range(3):
    plt.subplot(1,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.suptitle('Masking with Perimeter Contour', size=12, y=0.8)
plt.text(-520, 300, "*A mask is created from the perimeter contour and applied to the image.")
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
plt.show()

## Scale perimeter contour

# Contour of perimeter boundary is scaled by 0.98 relative to original centroid 
ret,thresh_bad = cv.threshold(result_bad,190,255,cv.THRESH_BINARY)
ret,thresh_f = cv.threshold(result,190,255,cv.THRESH_BINARY)

# plot 
titles = ['Uncorrected', 'Scaled by 0.98', 'Spurious non-zero pixels', 'Pixels removed']
images = [final_conts_img, scale_ex, thresh_bad, thresh_f]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.suptitle('Perimeter contour tightened to correct mask', size=12, y=1.01)
plt.text(-350, 300, "*The contour of the agar perimeter was scaled by 0.98 \nrelative to "\
        "the centroid to remove spurious pixels")
plt.subplots_adjust(left=0.1, bottom=0.1, right=None, top=None, wspace=0.01, hspace=0.3)
plt.show()

Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


########################################
## Experimental Results
########################################

## Functions

######################################
# Image processing

#Image to greyscale
def get_gs(path):
    gs = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return gs

#Blur image
def get_blur(image, value=9):
    blur = cv.GaussianBlur(image, (value,value), 0) #Gaussian blur applied 
    return blur
        
#Image to binary Gaussian thresholding
def get_gaus(image):
    thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 23,3)
    return thresh
    
#Image to binary Otsu thresholding
def get_otsu(image):
    blur = cv.GaussianBlur(image, (7,7), 0) #Gaussian blur applied 
    (T, thresh ) = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    return thresh
    
#Image to binary from path
def get_binary(path):
    gs = cv.imread(path, cv.IMREAD_GRAYSCALE)
    ret, thresh = cv.threshold(gs,200,255,cv.THRESH_BINARY)
    return thresh

#Image to binary from array
def get_binary_from_array(image, threshold_value=190):
    ret, thresh = cv.threshold(image, threshold_value, 255, cv.THRESH_BINARY)
    return thresh

#Get grand plate binary
def get_grand(image):
    contrasted_img = cv.equalizeHist(cv.GaussianBlur(image, (19,19), 0)) 
    grand = get_binary_from_array(contrasted_img, 110)
    return grand

#Save binary images
def save_binary(images_dir):
    os.chdir(binary_dir)
    #Loop through images
    for image in os.listdir(images_dir):
        path = os.path.join(images_dir, image)
        binary_img = get_binary(path)
        binary_result = Image.fromarray(binary_img)  
        #Save binary image
        with binary_result as f:
            f.save(Path("binary_" + image).stem + '.png')
    os.chdir(r"C:\Users\olive\Documents\QAC 307\images")

######################################
# Contours

#Get external contours
def get_ext_conts(image):
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy    

#Get internal contours
def get_conts(image):
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

#Find grand countour
def get_grand_cont_area(contours):
    lst = []
    for c in contours: # Get contour with greatest area
        area = cv.contourArea(c)
        lst.append(area)
    area_max = max(lst)
    return area_max

#Smooth contour
def smooth_conts(contours):
    smoothened = []
    for contour in contours:
        x,y = contour.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        tck, u = splprep([x,y], u=None, s=1.0, per=1)
        u_new = np.linspace(u.min(), u.max(), 25)
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
        smoothened.append(np.asarray(res_array, dtype=np.int32))
    return(smoothened)

#Draw contours
def draw_conts(image, target_img, retr="tree"):
    if retr == "ext":
        contours, hierarchy = get_ext_conts(image)
        cv.drawContours(target_img, contours, -1, (0,255,0), 1)
    elif retr == "tree":
        contours, hierarchy = get_conts(image)
        cv.drawContours(target_img, contours, -1, (0,255,0), 1)

#Find agar perimeter
def get_bounds(image):
    blur = cv.GaussianBlur(image, (15,15), 0)
    ret, thresh = cv.threshold(image,185,255,cv.THRESH_BINARY)
    #thresh = get_gaus(image) # Gaussian image
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Perimeter contour
    lst = []
    for c in cnts: # Get contour with greatest area
        area = cv.contourArea(c)
        lst.append(area)
    area_max = max(lst)
    return cnts, area_max

#Scale contour function
def scale_contour(cnt, scale):
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)
    return cnt_scaled
#Draw perimeter contour
def draw_bounds(target_img, contours, scale=False):
    bound_area = get_grand_cont_area(contours)
    # Loop through contours and draw largest as perimeter
    if scale == True:
        for c in contours:
            area = cv.contourArea(c)
            if area == bound_area:
                cv.drawContours(target_img, [scale_contour(c, 0.98)], -1, (255,0,0), 1)
            else:
                pass
    elif scale != True:
        for c in contours:
            area = cv.contourArea(c)
            if area == bound_area:
                cv.drawContours(target_img, [c], -1, (255,0,0), 1)
            else:
                pass

#Approximate contour
def approx_cont(contours):
    cnts = imutils.grab_contours(contours)
    c = max(cnts, key=cv.contourArea)
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.0001 * peri, True)
    return approx
            
#Fit ellipse to contour
def draw_ellipse(target_image, contours):
    color = [255, 255, 255]
    minEllipse = [None]*len(contours)
    ellipse_area = [None]*len(contours)
    result = -1
    
    for i, c in enumerate(contours):
        area = cv.contourArea(c)
        if area <= 100:  # skip ellipses smaller then 10x10
            continue
        if c.shape[0] > 2500:
            ellipse = cv.fitEllipse(c)
            x = ellipse[0][0]  # center x
            y = ellipse[0][1]  # center y
            angle = ellipse[2]  # angle
            a_min = ellipse[1][0]  # asse minore
            a_max = ellipse[1][1]
            a = int(a_max / 2)
            b = int(a_min / 2)
            ecc = (np.sqrt((a**2) - (b**2)) / a)
            if ecc < 0.3:
                minEllipse[i] = ellipse
                ellipse_area[i] = (ellipse[1][0]/2.0) * (ellipse[1][1]/2.0) * math.pi
                
    min_area = min([i for i in ellipse_area if i != None])
    
    for i, c in enumerate(contours):
        if ellipse_area[i] == min_area:
            result = minEllipse[i]
            cv.ellipse(target_image, minEllipse[i], color, 2)
        
            return result






######################################
# Masking

#Apply mask to image
def apply_mask(image, mask_contour, scale=1.0):
    color = [255, 255, 255]
    stencil =  np.zeros(image.shape, image.dtype)
    bound_area = get_grand_cont_area(mask_contour)
    for c in mask_contour:
        area = cv.contourArea(c)
        if area == bound_area:
            cv.fillPoly(stencil, [scale_contour(c, scale)], color)
        else:
            pass
    result = cv.bitwise_and(image, stencil)
    return result

#Get masked plate image
def get_masked_grand(image, scale=1.0):
    masked = image
    grand_binary = get_grand(masked)
    mask_cont = get_conts(grand_binary)[0]
    result = apply_mask(masked, mask_cont, scale)
    return result
    
#Ellipse mask
def apply_ellipse_mask(ellipse, image):
    color = [255, 255, 255]
    stencil =  np.zeros(image.shape, image.dtype)
    x = ellipse[0][0]  # center x
    y = ellipse[0][1]  # center y
    angle = ellipse[2]  # angle
    a_min = ellipse[1][0]  # asse minore
    a_max = ellipse[1][1]
    a = int(a_max / 2)
    b = int(a_min / 2)
    a = int(a*0.98)
    b = int(b*0.98)
    poly = cv.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (b, a), int(ellipse[2]), 0, 360, 5)
    cv.fillPoly(stencil, [poly], 255)
    result = cv.bitwise_and(image, stencil)
    return result

#Get final masked binary images    
def mask_binary(image, binary):
    masked = get_masked_grand(image, scale=0.95) #Grand plate perimeter
    blur = get_blur(get_binary_from_array(masked, threshold_value=200), value=15) #Gaussian blur and binary thresholding
    blur = cv.equalizeHist(blur) #Contrast
    gaus = get_gaus(blur)
    conts = get_conts(gaus)
    ellipse_array = draw_ellipse(image, conts[0])
    result = apply_ellipse_mask(ellipse_array, binary)
    return result
    
    
#Save masked binary
def save_masked_as_binary(lst):
    os.chdir(r"C:\Users\olive\Documents\QAC 307\images\images_binary_masked")
    for i,img in enumerate(lst):
        binary_img = get_binary_from_array(img, threshold_value=205)
        binary_result = Image.fromarray(binary_img) 
        with binary_result as f:
            f.save(Path("binary_" + f"{i+1}").stem + '.png')

    
#############################
## Area calculations

#Calculate area from non-zero pixels
def get_area(arr):
    area = cv.countNonZero(arr)
    return area    
        
        
#######################################
## Area calculation
#######################################
## Directories 

#Images
img_dir = r"C:\Users\olive\Documents\QAC 307\images\images_data" #Unmodified images
binary_dir = r"C:\Users\olive\Documents\QAC 307\images\images_binary_masked" #Binary images
area_lst = []
for i in range(1,19):
    img = f"binary_{i}"
    path = os.path.join(binary_dir, f"{img}.png")
    arr = get_binary(path)
    area = get_area(arr)
    area_lst.append(area)
    print(area, img)
5072 binary_1
13098 binary_2
20247 binary_3
16500 binary_4
6695 binary_5
16438 binary_6
23926 binary_7
11473 binary_8
35428 binary_9
17890 binary_10
10758 binary_11
33088 binary_12
25549 binary_13
20358 binary_14
21955 binary_15
48781 binary_16
66536 binary_17
30833 binary_18
###############################
## Make figures
###############################
control_path = r"C:\Users\olive\Documents\QAC 307\images\control"
uv_high_path = r"C:\Users\olive\Documents\QAC 307\images\uv_high"
uv_low_path = r"C:\Users\olive\Documents\QAC 307\images\uv_low"
control_images = []
for file in os.listdir(control_path):
    path = os.path.join(control_path, file)
    img = cv.imread(path)
    control_images.append(img)
control_images[0].shape
(1666, 1960, 3)
uv_high_images = []
for file in os.listdir(uv_high_path):
    path = os.path.join(uv_high_path, file)
    img = cv.imread(path)
    uv_high_images.append(img)
uv_low_images = []
for file in os.listdir(uv_low_path):
    path = os.path.join(uv_low_path, file)
    img = cv.imread(path)
    uv_low_images.append(img)
cap = 'Control Plates'

images = control_images
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i][0:1600, 0:1800])
    plt.xticks([]),plt.yticks([])
plt.suptitle(cap)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()

cap = 'UV High Plates'

images = uv_high_images
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i][0:1600, 0:1800])
    plt.xticks([]),plt.yticks([])
plt.suptitle(cap)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()



cap = 'UV Low Plates'

images = uv_low_images
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i][0:1600, 0:1800])
    plt.xticks([]),plt.yticks([])
plt.suptitle(cap)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()


 
6. References
Pereira, R. V., et al. (2014). "Evaluation of the effects of ultraviolet light on bacterial contaminants inoculated into whole milk and colostrum, and on colostrum immunoglobulin G." J Dairy Sci 97(5): 2866-2875.
	
Pullerits, K., et al. (2020). "Impact of UV irradiation at full scale on bacterial communities in drinking water." npj Clean Water 3(1): 11.
	

