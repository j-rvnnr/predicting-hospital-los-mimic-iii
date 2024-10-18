# Introduction
The Length of stay (LOS) is a key metric within the healthcare system. The amount of time a patient stays in hospital affects hospital efficiency, patient outcomes and financial operations. In most hospitals, predicting LOS is key for smooth functioning of a hospital, with shorter stays often associated with improved patient flow and resource allocation, while extended hospital stays have the opposite effect.  The goal of this project was to develop a system using machine learning to accurately predict the length of a hospital stay using the MIMIC-III dataset. (A freely available dataset of hospital information)

Globally, healthcare expenditure accounts for a significant portion of national GDP's, with the average being around 9.3% in OECD countries.

![Pasted image 20241015185743](https://github.com/user-attachments/assets/04b13197-0b5f-440e-8f78-c8242a25d608)


While hospitals aim to improve the outcomes for patients, keeping costs down and efficiency up are other goals of the healthcare system. There is no single standard method to predict the length of stay of patients, with some hospitals using methods like taking the median across the hospital or care unit, and applying that across all patients, and some instead using a prediction made by a clinical team involved in each patient. Although these methods 'work', it is clear that there is room to improve. 

My first step was looking into different methods of prediction which has been performed in the past. Methods used include Random Forests (Elbattah and Molloy, 2017) and deep neural networks (Bishop et al, 2021). Both of these methods are robust, but I wished to improve upon them in my work. 

My aim therefore is:
- Accurately predict the LOS of hospital patients to a more accurate degree than simply using the median value of the stays. 
- Use Artificial Neural Networks to categorise patients into short, medium and long stay groups, and predict their precise LOS from there.
___
# Data
This project used data from the MIMIC-III dataset, a large, comprehensive, and (mostly) free and publicly available dataset. It includes data on over 46,000 patients admitted to the Beth Israel Deaconess Medical Center between 2001 and 2012. The data includes demographic, hospital and billing data, and we can take these points to provide a full picture of someone's hospital stay. The data is also de-identified in order to preserve patient privacy, however none of the de identification measures will cause us problems. 

All of the data is made up of patients who spent some time in an intensive care unit (ICU), however it does not only include data from the ICU.



## pre processing
When pre-processing the data there were a few concerns that we had to address.
- No data from children between 5 and 16
- Patients ages above 90 years had a value of 300 added to them
- Patients who died in hospital

### Child data
Children pose 2 challenges in this project; firstly, their ages are not recorded as a matter of privacy between the ages of 5 and 16 *(unsure if this range is correct-check)*, and children under 5 who spend time in the neonatal unit have a very different pattern of LOS than the rest of the patients. For this reason, the data was discarded, and instead we will focus on adults, 16 +.

### Older patients
In order to preserve the privacy of patients over 90 years of age, their ages were set to 300 years old. We can use this knowledge to filter these patients into a 90+ age category. Whilst knowing their exact age would be helpful, it is hoped that the category of people aged 90+ is small enough and contains enough information that more granular data is not required. 

### Patients who died
This is the simplest of all three of these pre-processing groups. These patients were removed, as their LOS data (derived from when they entered hospital until they died) is unhelpful for our purposes. We only wish to predict the LOS of a patient who 'completes' a hospital stay, and therefore, shall be excluding any patient who died. 


![Pasted image 20241016184210](https://github.com/user-attachments/assets/c5fcc1fb-c0d4-47a9-b36b-d87ae308549a)
*note that patients who stayed longer than 30 days were added to the 30 day bin*

## Data Exploration
As shown in the histogram above, it's fairly easy to calculate some basic information about the hospital visitors. The data is heavily skewed, with a median stay of 6.9 days and a mean stay of 8.9 days. The median stay already has a large error when compared to the mean stay, so there is some room for accurate predictions.

___
# Methodology
## Filtering and cleaning
The first step in carrying out the project was filtering and cleaning the data. We take the pre-processed data, and we performed some transformations to give greater context to the hospital stays. For example, LOS was calculated in hours for increased precision, and additional variables such as 'time since admission' were added. 

## Statistical Analysis
Each feature in the data was analysed using either the Spearman Correlation Coefficient (numerical data) or the Kruskal-Wallis test (categorical data). The results of these tests were used to select which data points should be used to construct further features to improve the prediction model. All variables which were shown to have significance (p value below 0.05) were chosen at this point, to be further refined. This was admittedly a fairly blunt and imprecise method, however this method was chosen for speed, and due to the large number of features present in the data. 

Some variables such as religion and ethnicity were consolidated down to a few simpler values to reduce dimensionality, and a focus was made for adding numerical transformations of data, such as mean values and counts in order to increase the breadth of data available without increasing the dimensionality by too much.

## Prediction Models
### Classification model
Part of the theory of this project, was to split the patients into Short, Medium and Long stay patients, defined as <7 days for short stay, 7-21 days as medium stay and >21 days for long stay. 

The purpose behind this prediction model was to more accurately predict the LOS for the different classes of patient; with the hypothesis that these different classes of patient will have significantly different length of stay patterns. There was also initially a hope that the different predicted classes could be used in order to be used as a feature in the LOS prediction itself, however it was found to be not useful.

![Pasted image 20241017142606](https://github.com/user-attachments/assets/470e211f-8bb6-499f-a0e4-78ea4964e0ee)

The network structure was designed using a grid search for parameters such as the layer size and dropout value. The activation used was Parametric ReLU, to add another learned parameter to the system. 
### Regression Model
The main part of the prediction was a regression model, designed to predict the LOS time in hours of each patient. The structure is the same as the classification model, with a single node in the output layer instead for this regression task. I used Huber loss on this model, as it combines the best aspects of Mean Square error and Mean Absolute error. This helps manage outliers and improve prediction accuracy, as most patients fall within a narrow range, with a few being extreme outliers. 

## Notes on Model Application
#### A note on categorical/numerical data
During the course of this project, I came to the realisation that using the full data with both categorical and numerical features proved to calculate over a very long time per epoch (around 2 minutes), whereas using just the numerical data proved to be much faster (1.5 seconds per epoch on average, over 100x faster!) without a significant loss in accuracy of around 1.5% reduced error over 100 epochs. It was therefore decided that I would only use the numerical data for the models in the final version of the project, as it was sufficiently accurate and much faster. 
#### A note on time windows

when predicting whether a sample belongs to each stay class, the data was assembled using 'time windows', where the data obtained after the 'time window' was discarded, in an attempt to make the predictions more practical and realistic to a real world use case. The time windows chosen for this project were 24, 48 and 72 hours, and the predictions were made for all combinations of these time windows. They will be discussed more later, as using this method did raise some concerns, which should be corrected in a future application of this project. Essentially, whenever I mention something like '24 hour time window' what it means is 'all data taken from the first 24 hours of a patients hospital visit'.

___
# Results

## Classifier Results
One of the objectives of the project was to use the results from the classifier as a feature for the LOS prediction. The model was 87% accurate in predicting the classes of patients, however the model was significantly better at predicting short (0) and medium (1) stay patients than long (2) stay patients. 

![Pasted image 20241017204316](https://github.com/user-attachments/assets/3a0380c6-f161-41b7-90e7-37fedea06432)

the results for 48 and 24 hour time windows were also promising, with the 48 hour time window showing similar results to the 72 hour time window. The 24 hour time window was significantly worse than the others. 
![Pasted image 20241017204331](https://github.com/user-attachments/assets/245d6b60-296c-4686-9cbb-b6895be22f5b)
![Pasted image 20241017204412](https://github.com/user-attachments/assets/1af10236-d390-4b29-9d77-609210db4b57)

The ROC curve for the classifier results was promising too, however the area under curve is probably over represented in class 2, as it was calculated vs the total of the other 2 classes, and as there were significantly fewer entries in class 2, it gave a very high result. 
![Pasted image 20241018204356](https://github.com/user-attachments/assets/7d5fd2b8-77ca-4d10-bf8c-d5ad54a58e9f)


## Regressor Results
The regressor model provided fairly positive results too, however there is a noticeable anomaly when viewing the results: a hard cut-off line was present and visible on the plotted results, which only seems to affect long term patients. 
![Pasted image 20241018204608](https://github.com/user-attachments/assets/fbe93c5a-9bed-4d80-833c-2e2adb481971)

The success of the model was judged by the mean absolute error, which should be minimised using the Huber loss function. Taking a time window of 48 hours gives a result of around 40 hours of error across the entire dataset, considering all 3 classes, however if we exclude the long stay patients, the same model gives an error of 29 hours, an improvement of 27.5% over the full dataset. If we further reduce to just calculate the predicted short stay patients, the error is reduced to 14 hours, a 65% improvement over the full data, and a 51.7% improvement over the data with long stays excluded. 
![Pasted image 20241018205453](https://github.com/user-attachments/assets/c7f529f8-9137-4fe3-a9e1-4c2d2b77e06e)

![Pasted image 20241018205458](https://github.com/user-attachments/assets/c56782b5-629f-4409-8994-1be7db88d20b)

____
# Discussion
## Results discussion
### Using Classification Results as a feature
The goal of using classifier results as a feature to enhance the predictive capability of the model failed, as the model managed to delineate between the different stay classes by itself. Whether these classes were or were not considered was shown to be largely irrelevant, however the presence of these predicted classes did allow further exploration within the results, such as splitting the results based on the predicted stay class and determining the error from that. 

### Accuracy
The model proved accurate as a method to predict hospital stays. As there is no universal standard for predicting the length of a hospital patients stay, it is difficult to quantify how successful this project was, however, the relatively small error within the short and medium stay patients are strong results. This suggests that a good method for utilising this model would be a hybrid approach, where first the patient would be sorted into predicted stay class, and then if the patient is a short or medium stay patient, they could have an accurate time prediction made, and if they are long stay, perhaps using the median LOS is sufficient. 

## Limitations
## The prediction ceiling
The model shows a false "cap" around 1400 hours in the prediction, even though no such limit was imposed on the model. This issue persisted through different loss functions (MSE and MAE). Initially I thought it could be due to the stay classes, and the model extrapolating that there were more than just the 3 given stay classes, however when stay classes were not considered, the cap remained. This was a rather annoying error and definitely doesn't help the results, so in the future, investigating and eliminating this cap would be a strong step in refining this model.

## Data Available at Admission
Part of the self-imposed challenge was to use only data that was available for a certain time after admission, in the case of this project that was 48 hours. Some issues come from the fact that a few of the types of data, such as specialised diagnosis codes, were only provided after the patient left hospital. A more rigorous data cleaning process with a focus on accuracy and more derived features would allow this prediction to improve. 

## Future Research
Future work I would like to explore include:
- Ensemble machine learning using a combination of this ANN, and alternatives such as gradient boosting. 
- Using the actual stay classes instead of the predicted ones. This one seems obvious now, but it was not when I was performing the test.
- A more temporal approach. Many of the data points have exact time measurements given, it may be helpful to build a time series for each patients stay, and see if that has any significant impact on prediction. 
