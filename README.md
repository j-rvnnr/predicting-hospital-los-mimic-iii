# Predict-Hospital-LOS
Predicting the Length of Stay of hospital patients using the MIMIC- III Dataset.

# 0: Overview
## 0.1: Why?

It's important to predict hospital length of stay (LOS, los) for a number of reasons, such as improving patient flow, Increasing healthcare efficiency (both inpatient and outpatient), saving money in the hospital system, and improving outcomes for patients.

This project is my masters thesis project for my Data Science postgrad at the University of Aberdeen. This is not the thesis itself, but instead supplementary information. I will try to keep a degree of professionalism here, but this file explaining is likely to be significantly more casual than any academic writing I do on the topic.

## 0.2: What?

The mimic -III dataset is a popular dataset for studies on hospital patients as it's large, broad, and freely accessible (mostly). There are a large number of different tables, which include numerous datapoints from all across the hospital. We will be using and selecting a wide variety of these data to use in our experimentation,
and hopefully produce a lean, fast and effective machine learning model which will be able to somewhat accurately predict an  unseen patients los.

The plan is to use a 2 stage machine learning approach, first to identify outliers within the data, then to predict the length of stay of patients within those groups. (outliers and non-outliers)

## 0.3: How?

This overview serves as a brief plan on how I wish to carry out the process. Much of this is constructed from notes taken during the undertaking of the project, although it is compiled once the majority of the project work is complete. 

The process can be split into 3 easy sections. 
- data exploration
- feature engineering
- machine learning

below, I will describe how I completed each of these steps, in the form of an article/report about the project. 

## 0.4: Insights

This is the first project of this scope I have undertaken, and as a result, it was a bit sloppy. The project was also not what I expected coming in. Before starting, I expected this project to be 75% machine learning, however once I learned how simple it was to set up a neural network in python with Keras/Tensorflow, I realised that this project was actually mostly about data management and filtering and feature engineering. This didn't dtract from the project at all, and instead, made it rather fun, if I do say so! 