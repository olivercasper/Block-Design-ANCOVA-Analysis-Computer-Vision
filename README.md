# Randomized Block Design: Quantifying Treatment Effect with Computer Vision 

Experimental results from image data using R and OpenCV2 for Python 










1. Introduction


	Bacteria are easily the most abundant organisms on Earth, representing one of the three basic domains of life. These single-celled prokaryotes are found nearly everywhere and serve a diverse set of biological roles, ranging from symbiotic to pathogenic. As such, bacterial contamination poses a threat to public health which has been addressed in supply chains. One interesting approach to sterilization is the application of ultraviolet (UV) irradiation.
 Exposure to ultraviolet light damages the structure of DNA molecules, preventing mitotic replication of bacterial cells. Researchers have demonstrated the efficacy of UV irradiation in the sterilization of both dairy products (Pereira, Bicalho et al. 2014) as well as drinking water (Pullerits, Ahlinder et al. 2020). Our study aimed to assess the relationship between UV irradiance and relative extent of bacterial growth in agar medium. We presently discuss the experimental and statistical design we employed to explore the question: “How does the extent of bacterial growth over an incubation period vary with level of initial ultraviolet irradiation?” 







2. Methods



  ![image](https://github.com/olivercasper/Block-Design-ANCOVA-Analysis-Computer-Vision/assets/135169906/469f8220-1a11-4747-96cb-b1db881b33cb)



2.1 Experimental Design

2.1.1 Sample
	Bacteria were grown in a series of petri dishes, with each plate representing one observational unit. A nutritionally rich agar medium made with Luria broth was chosen here as the growth substrate. Initial bacterial inocula were collected from saliva using sterile cotton swabs and subsequently transferred to agar. Treatment was evenly and randomly assigned to units throughout the inoculation process through a randomized block design with order as the blocking variable. There were 6 levels of order representing inoculation from start to finish (3 units per level). In total, 18 inoculated plates were used in our sample, with 6 plates representing each level of UV treatment (N = 18; n = 6). 


2.1.2 Procedure
	To investigate the relative effect of UV irradiation on bacterial growth, plates were exposed to 3 different intensities of UV light for 45 seconds prior to a 3-day incubation period. The levels of treatment were either UV high, UV low, or no exposure (control). Plates assigned to UV high were irradiated at half the vertical distance (11 cm vs. 22 cm) . The UV low group therefore received a dose that was ¼ of the intensity of UV high due to the inverse square law. Plates of both treatments were arranged in 3x2 arrays on separate shelves and the no-light control plates were placed in a drawer. A single UV lamp, set to “high”, with two adjustable arms was used as the light source for both intensities. Following incubation, plates were photographed at the same distance and disposed of. The value of total bacterial growth was then calculated for each observation using a computer vision model. 


2.2 Computer Vision
	To approach the problem of accurately quantifying the growth response, we relied on various computer vision techniques to process and analyze the images of our plates post-incubation. The aim here was to count the number of pixels representing bacterial colonies and record that value as the growth response while ignoring all other pixels. We relied on the OpenCV library for Python 3.10.7 along with NumPy and Pillow libraries for the entire computer vision procedure. 
Our methods here involved two main components: thresholding and edge detection. Binary thresholding (Figure 1) was used to represent each image as a 2-dimensional binary matrix where the sum of elements yields the value of growth. Adaptive gaussian thresholding was used separately to prepare images for edge detection by maximizing the contrast between boundaries of discrete objects within the image (Figure 2). Contours were fit to the gaussian image and filtered for shape and area to approximate an ellipsoid mask which was used to remove non-zero pixels beyond the agar boarder in the corresponding binary image. The final result was a set of binary plate images containing only non-zero bacteria pixels (Figure 3). These results were then passed to OpenCV’s countNonZero() function to obtain the integer value of growth for each observation.   

Figure 1: Binary thresholding filter






![image](https://github.com/olivercasper/Block-Design-ANCOVA-Analysis-Computer-Vision/assets/135169906/8f50c225-f05f-4918-8a34-63b37c0cddbc)









Figure 2: Binary and adaptive Gaussian thresholding applied to greyscale image


![image](https://github.com/olivercasper/Block-Design-ANCOVA-Analysis-Computer-Vision/assets/135169906/465c980c-1944-45b7-8331-ec6161d736a9)




Figure 3: Unwanted pixels are removed with masking image in shape of perimeter contour. Here the contours of the agar perimeter are fit in the Gaussian image with OpenCV’s canny edge detection to create an ellipsoid mask. The adaptive Gaussian filter provides the boundary contrast necessary for the machine learning model to accurately determine the contours and hierarchy of objects within the image. 



![image](https://github.com/olivercasper/Block-Design-ANCOVA-Analysis-Computer-Vision/assets/135169906/5a2660d9-79fc-472b-abb0-206137a6dbdb)








2.3 Statistical Methods

2.3.1 Measures
	Data were recorded for the assignment of treatment, order block, array position, and growth. The natural log of growth was used as the response variable in analysis. Treatment was coded as either “uv_high”, “uv_low”, or “control”. Position in inoculation order was coded with integers 1:6 as an ordinal variable. Row and column values for array position were recorded for UV treatments and coded with integers 1:3 and 1:2 respectively, with “3” being the bottom-most row and “2” being the right column. Control plates were assigned a value of “0” for row and column positions. The growth response was coded as a discrete quantitative variable with each unit increase representing a single non-zero pixel of the final result image. 

2.3.2 Analysis
	We used analysis of variance with the inclusion of covariates (ANCOVA) to determine how the extent of bacterial growth varies among the three levels of UV treatment. Our model is represented by the equation:


![image](https://github.com/olivercasper/Block-Design-ANCOVA-Analysis-Computer-Vision/assets/135169906/204ce601-6599-438d-99da-ebfa287b74ab)


where τ, δ, γ, and φ are estimated effect terms of treatment, order, row position, column position respectively. Tukey’s multiple comparisons post-hoc was used to find the significant pairwise differences in effect across the 3 levels of treatment. All analysis was performed in R. The ggplot2 and gmodels R packages were used. 








3. Results  
	In line with our expectations, analysis of variance revealed overall difference in mean growth among UV treatments (F = 11.8 ; p = 0.0057) (figure 4). At α = 0.05, we reject the null hypothesis in favor of the alternate that at least one treatment mean varies. Tukey’s multiple comparisons post-hoc revealed that uv_high had a significantly lower value of log growth than both control (µhigh – µc = -0.71; p = 0.006) and uv_low (µhigh – µlow = -0.55; p = 0.02). There was no significant difference in log growth between uv_low and control (µlow  – µc = -0.15; p = 0.6). The blocking variable for inoculation order was also found to be significant (F = 11.4; p = 0.003) with growth displaying a positive trend with place in inoculation order (figure 5). Lastly, growth varied significantly among row positions (F = 6.2; p = 0.028) (figure 6). 

Figure 4: ln(growth) by treatment


![image](https://github.com/olivercasper/Block-Design-ANCOVA-Analysis-Computer-Vision/assets/135169906/615df120-800f-4169-9ba9-6d13f6083835)



Figure 5: ln(growth) by order block 


![image](https://github.com/olivercasper/Block-Design-ANCOVA-Analysis-Computer-Vision/assets/135169906/fd517425-5c2b-4499-b519-63d09180d175)




Figure 6: ln(growth) by row position


![image](https://github.com/olivercasper/Block-Design-ANCOVA-Analysis-Computer-Vision/assets/135169906/daef5714-2aa4-4667-a7c8-2f61b44354b1)




4. Discussion
	Based on the significant variance in mean growth between UV treatment levels (F = 11.8 ; p = 0.0057), we conclude that our experiment was mostly effective at meeting our endpoint. While the results do demonstrate the relationship of interest, the other significant factors in our model reveal flaws in procedural design and execution. The uneven growth across inoculation order indicates that transfer of bacterial material to the agar was not properly controlled. A future attempt of this experiment might consider using a volumetric inoculant instead of swabs. Furthermore, the significantly greater growth of the 3rd row suggests unequal irradiance of plates under the UV source. A square-shaped position array under a UV point source could possibly control relative irradiance here. Additionally, our statistical approach could be revised to balance position of replicates with row as a second block in a Latin square design. 
	Overall, we are pleased with the outcome of our experiment. In the present study, we set out to design experimental methods which would illustrate the impact of UV exposure on bacterial proliferation. We were interested to find not only this relationship, but also additional considerations for experimental design. The results of this analysis were a valuable lesson in unexpected outcomes and how data can reveal underlying experimental errors. 
 
 

 
5. References

Pereira, R. V., et al. (2014). "Evaluation of the effects of ultraviolet light on bacterial contaminants inoculated into whole milk and colostrum, and on colostrum immunoglobulin G." J Dairy Sci 97(5): 2866-2875.
	
Pullerits, K., et al. (2020). "Impact of UV irradiation at full scale on bacterial communities in drinking water." npj Clean Water 3(1): 11.
	

