In this file, I provide a general description about each of the files found in this repository,
the goal of each one, the algorithms used and other relevant information.

-CountriesGDPPerCapita_LinearRegression: this example uses LinearRegression by sklearn to predict the
GDP Per Capita of a list of 195 countries. First, I do some data cleaning, and remove
characters that can not be converted to float values, such as '$' and '%'. Next, I calculate
the GDP per capita by dividing the GDP column by the Population column for each country. 
After that, I create the X dataset by removing the target variable 'GDP Per Capita', and
the Y dataset which includes only the target variable. Afterwards, I split the data in training
and test, using a 80-20 split respectively, initialize the linear regression model, train it
and test it with the test dataset. For evaluation I use the R-square score. I also include
the coefficients of each feature, to see which is more significant when predicting a countrie's
GDP per capita.

-HeartDisease_SVC: here I use a Support Vector Classifier (SVC) with a linear kernel to perform
binary classification using a dataset with patient symptoms and labels for heart disease or no 
heart disease. First, I set the X and y datasets with features and target variable respectively.
I use pandas get_dummies for one-hot encoding. Later on, I initialize the SVC classifier and 
perform cross validation, followed by splitting 80-20 for train-test and calculating the accuracy
of the SVC. The accuracy score and the cross validation mean are very close to each other, which
means that this specific splitting corresponds to a mean accuracy, which is good. Following this, 
I calculate weighted metrics: recall, precision and F1-score, obtaining good results for each.
Next, I plot the confusion matrix and calculate more metrics by hand, including: Error rate, 
True positive rate, Specificity, Precision, Mathews correlation coefficient, Normalized Mathews 
correlation coefficient and F1 score. Lastly, I plot the ROC and PR curves, including the area
under the curve (AUC) for each, and diagonal lines showing the performance of a random classifier.
For both plots, the SVC shows predictive capacities better than a random classifier, and 
achieves satisfactory values for both AUCs. The performance of this SVC classifier is compared
with a Boosted Decision Tree with this same dataset in the next example.

-HeartDisease_BDT: in this implementation I use a Boosted Decision Tree (BDT) with the same
heart disease dataset for comparison with the SVC. I followed the same procedure as with the SVC,
but the BDT requires aditional hyper-parameter setting, such as the number of iterations per 
prediction, the learning rate per iteration and the maximum depth allowed for each decision tree.
I used 100 iterations, a 0.1 learning rate and the default maximum tree depth of 3 branches.
After performing cross validation, I calculated the same metrics as with the SVC, but obtained 
much better results, including an accuracy of 0.99, not that far of form the cross validation mean
of 0.97. The confusion matrix shows an outstanding performance, as can be confirmed by the additional
metrics calculated, such as Error rate, True positive rate, Specificity, Precision, Mathews 
correlation coefficient and F1 score. After that, I plotted the ROC and PR curves, and obtained
results showing an exceptional capacity for this classifier, with AUC values of 0.99 for both, 
vastly outperforming the SVC classifier. Additionaly, I implemented a method for calculating the
uncertainty in the predictions of the BDT, based on using the staged probabilities for each 
prediction to calculate a standard deviation in the probability of each prediction, and using
that standard deviation to impose upper and lower limits in the number of positive and negative 
heart disease results.

-Iris_SVC: for this dataset, a 3 category classification task is achieved using the Support
Vector Classifier with linear kernel. The usual procedure is followed for the classification,
using a 3 category confusion matrix and metrics such as accuracy, recall, precision and F1 to
evaluate. The SVC has an outstanding performance with this dataset. I also use this example
to showcase useful plots such as: pair plot, box plot, violin plot, swarm plot and
correlation heatmap of the dataset features.     
 
 
