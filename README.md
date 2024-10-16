![Pasted image 20241016184210](https://github.com/user-attachments/assets/dafe8108-9fdb-4952-9195-8740c81fd883)# Introduction
The Length of stay (LOS) is a key metric within the healthcare system. The amount of time a patient stays in hospital affects hospital efficiency, patient outcomes and financial operations. In most hospitals, predicting LOS is key for smooth functioning of a hospital, with shorter stays often associated with improved patient flow and resource allocation, while extended hospital stays have the opposite effect.  The goal of this project was to develop a system using machine learning to accurately predict the length of a hospital stay using the MIMIC-III dataset. (A freely available dataset of hospital information)

Globally, healthcare expenditure accounts for a significant portion of national GDP's, with the average being around 9.3% in OECD countries.
![[Pasted image 20241015185743.png]]
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
The first step in carrying out the project was filtering and cleaning the data. We take the pre-processed data, and we performed some transformations to give greater context to the hospital stays. For example, LOS was calculated in hours for increased precision, and additional variables such as 'time since admission' were added. The data was also restructured so that all the chosen data was represented in each row, rather than spread across a large number of tables. 
