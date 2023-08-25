========================================================================================================
Short description:

Classification methods used: Support Vector Classifier (HeartDisease_SVC, Iris_SVC), 
Boosted Decision Tree (HeartDisease_BDT) and Fully Connected Neural Networks (Iris_NN_PyTorch, 
Iris_NN_TensorFlow). The metrics used for evaluation include: confusion matric, accuracy, specificity, 
precision, mathew's correlation coefficient, F1 score, ROC and PR curves with area under the curve
(AUC). Additional plots shown include: pair plots, box plots, violin plots, swarm plots and
correlation heat maps. 

The Neural Networks are implemented with PyTorch and TensorFlow, each including hidden layers,
specified activation functions for each and for the output layer, as well as accuracy and loss
function evlauation, and a confusion matrix to verify.

Linear regression is implemented with sklearn (CountriesGDPPerCapita_LinearRegression), including the 
study of some correlation between features. The R-squared score is used for evaluation, and the 
coefficients of each feature are shown to determine which are the most significant for the predictions.

Clustering is used as a method of unsupervised machine learning (KMeans_WineClustering). Inertia is 
used for plotting an elbow curve and determine the optimal number of clusters. The metric silhouette
is also used to evaluate the model. a pair plot is also included to look for a pair of features with 
a good separation between cluster labels. These features are used to obtain a scatter plot to show 
the separation between clusters. 

========================================================================================================
Detalied description of each file:

-CountriesGDPPerCapita_LinearRegression: This example uses LinearRegression by sklearn to predict 
the GDP Per Capita of a list of 195 countries. First, I do some data cleaning. Next, I calculate
the GDP per capita by dividing each country's GDP column by the Population column. 
After that, I created the X dataset by removing the target variable 'GDP Per Capita', and
the Y dataset which includes only the target variable. Afterward, I split the data in training
and test, using an 80-20 split respectively, initialize the linear regression model, train it,
and test it with the test dataset. For evaluation, I use the R-square score. I also include
the coefficients of each feature, to see which is more significant when predicting a country's
GDP per capita.

-HeartDisease_SVC: Here I use a Support Vector Classifier (SVC) with a linear kernel to perform
binary classification using a dataset with patient symptoms and labels for heart disease or no 
heart disease. First, I set the X and y datasets with features and target variables respectively.
I use pandas get_dummies for one-hot encoding. Later on, I initialize the SVC classifier and 
perform cross-validation, followed by splitting 80-20 for train-test and calculating the accuracy
of the SVC. The accuracy score and the cross-validation mean are very close to each other, which
means that this specific splitting corresponds to a mean accuracy, which is good. Following this, 
I calculate weighted metrics: recall, precision, and F1-score, obtaining good results for each.
Next, I plot the confusion matrix and calculate more metrics by hand, including Error rate, 
True positive rate, Specificity, Precision, Mathews correlation coefficient, Normalized Mathews 
correlation coefficient, and F1 score. Lastly, I plot the ROC and PR curves, including the area
under the curve (AUC) for each, and diagonal lines showing the performance of a random classifier.
The SVC shows predictive capacities better than a random classifier for both plots and 
achieves satisfactory values for both AUCs. The performance of this SVC classifier is compared
with a Boosted Decision Tree with this same dataset in the next example.

-HeartDisease_BDT: In this implementation, I use a Boosted Decision Tree (BDT) with the same
heart disease dataset for comparison with the SVC. I followed the same procedure as with the SVC,
but the BDT requires additional hyper-parameter settings, such as the number of iterations per 
prediction, the learning rate per iteration and the maximum depth allowed for each decision tree.
I used 100 iterations, a 0.1 learning rate and the default maximum tree depth of 3 branches.
After performing cross-validation, I calculated the same metrics as with the SVC but obtained 
much better results, including an accuracy of 0.99, not that far of from the cross-validation mean
of 0.97. The confusion matrix shows an outstanding performance, as can be confirmed by the additional
metrics calculated, such as Error rate, True positive rate, Specificity, Precision, Mathews 
correlation coefficient, and F1 score. After that, I plotted the ROC and PR curves and obtained
results showing an exceptional capacity for this classifier, with AUC values of 0.99 for both, 
vastly outperforming the SVC classifier. Additionally, I implemented a method for calculating the
uncertainty in the predictions of the BDT, based on using the staged probabilities for each 
prediction to calculate a standard deviation in the probability of each prediction and using
that standard deviation to impose upper and lower limits in the number of positive and negative 
heart disease results.

-Iris_SVC: For this dataset, a 3-category classification task is achieved using the Support
Vector Classifier with linear kernel. The usual procedure is followed for the classification,
using a 3-category confusion matrix and metrics such as accuracy, recall, precision, and F1 to
evaluate. The SVC has an outstanding performance with this dataset. I also use this example
to showcase useful plots such as pair plots, box plots, violin plots, swarm plots and
correlation heat maps of the dataset features. 

-Iris_NN_PyTorch: Here I classify the Iris dataset with a fully connected neural network that 
uses the ReLU activation function. For this, I use the PyTorch framework, including the use of
torch tensors. The NN has two hidden layers, and is trained for 1000 epochs, using the Cross
Entropy loss function and the Adam criterion for updating the weights, with a learning rate 
of 0.001. The NN is then evaluated with a correlation matrix and perfectly classifies the dataset.

-Iris_NN_TensorFlow: the Iris dataset is classified with another fully connected neural network,
this time using TensorFlow. The specifications of this FCNN are almost identical to the ones used
for PyTorch, two hidden layers with relu as activation function, and softmax for the output layer.
This time, only 100 epochs are used for training, but the same perfect classification results are
obtained.

-KMeans_WineClustering: here I use the unsupervised machine learning algorithm clustering. This 
dataset has features for wine from 3 different vineyards, the task is to cluster the wine 
instances and group each to the wineyard it came from. For this, the KMeans algorithm from sklearn
is used, along with the intertia metric, the elbow curve and the silhouette metric. Lastly, I 
include a pairplot to look for a pair of features that create a good separation between clusters
and do a scatter plot to show how each cluster is separated from the rest.    
========================================================================================================
