# Results
## Classifier Results
The classifier model is designed as a preliminary estimator for the regression model. We want to predict the 'Stay class' of each patient, as certain stay classes have better prediction outcomes from the regressor. The stay classes we chose were Short stay (< 7 days in hospital), Medium stay (7-21 days in hospital) and Long stay (21+ days in hospital). Although these selections seem arbitrary, there is no accepted definition for a short, medium, or long stay in hospital, therefore I have chosen the categories for this in a way that serves my model and data the best. In the following plots, Short stay is class 0, Medium stay is class 1 and long stay is class 2. The visualisation of the classifier results only takes the final pair's outcome, however, the predictions are applied and cross validated across the entire dataset. 

### Visualising Class Predictions


![confusion_matrix_abs_classifier-0731-1106](https://github.com/user-attachments/assets/2e25c041-e4c6-47bf-9097-5c4ee6cbbbc0)

This is the confusion matrix of the absolute values predicted by the classifier model. It can be seen here, that the classes are heavily imbalanced, with class 0 being the largest part of the test set, but also the most accurately predicted class. Class 2 on the other hand, appears to be worst predicted class. There is an overlap between class 1 and class 2. 

![confusion_matrix_percentage_classifier-0731-1106](https://github.com/user-attachments/assets/675beef8-b99d-42f5-b1cb-54027ab7566b)

In order to determine specifically which class is the most accurately predicted, this plot is also shown with the values represented as percentages. Class 0 is accurately predicted with 92.5% accuracy, class 1 with 86.6% accuracy and class 2 with 75.1% accuracy. We will use these results, to create a 'predicted class' variable within the data, which will be passed on to the regression model, to hopefully improve the prediction of that model. We can also filter the data to consist only of certain classes, potentially improving the model's predictive power further.

![model_loss_classifier-0731-1106](https://github.com/user-attachments/assets/6566e6c0-130c-4ef7-8589-88538c9a27b7)

The loss for the classifier shows that it stabilises quickly and remains stable throughout the training process. 

## Regression Results
The regression model takes the same data provided to the classification model, as well as the results of the classifier. The predictions of the regressor varies widely depending on which class is targeted for the regression. 
