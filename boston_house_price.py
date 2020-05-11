
# Import libraries necessary for this project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve as curves
from sklearn.model_selection import validation_curve as curvesvalidate
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
###########################################
from sklearn.datasets import load_boston

#The main dataset with 506 samples and 14 features
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
#boston.head()
boston['MEDV'] = boston_dataset.target
print(boston)
correlation_matrix = boston.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)

# Load the Boston housing reduced dataset of 489 samples and 4 features 
data = pd.read_csv('housing.csv')
print(data)
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
print("Boston housing dataset has %d data points with %d variables each" %(data.shape[0],data.shape[1]))


#description of the dataset
# Minimum price of the data
minimum_price = np.min(prices)
# Maximum price of the data
maximum_price = np.max(prices)
# Mean price of the data
mean_price = np.mean(prices)
# Median price of the data
median_price = np.median(prices)
# Standard deviation of prices of the data
std_price = np.std(prices)
# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: $%.2f" %(minimum_price))
print("Maximum price: $%.2f" %(maximum_price))
print("Mean price: $%.2f" %(mean_price))
print("Median price $%.2f" %(median_price))
print("Standard deviation of prices: $%.2f" %(std_price))
      

# Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)
print("Training and testing split was successful.")

#create regression line for training and testing dataset
plt.figure(figsize=(20, 5))
for i, col in enumerate(features.columns):
    plt.subplot(1, 3, i+1)
    x = X_train[col]
    y = y_train
    plt.plot(x, y, 'o')
    # Create regression line
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('prices')
    plt.show()

plt.figure(figsize=(20, 5))
for i, col in enumerate(features.columns):
    # 3 plots here hence 1, 3
    plt.subplot(1, 3, i+1)
    x = X_test[col]
    y = y_test
    plt.plot(x, y, 'o')
    # Create regression line
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('prices')
    plt.show()

# Produce learning curves for varying training set sizes and maximum depths
#Analyze the model performance with different max_depth
def ModelLearning(X, y, clf):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing scores for each model are then plotted. """
    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], test_size = 0.2, random_state = 0)
    # Generate the training set sizes increasing by 50
    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)
    # Create the figure window
    fig = plt.figure(figsize=(10,7))
    # Create three different models based on max_depth
    for k, depth in enumerate([1,3,6,10]):
        # Create a Decision tree regressor at max_depth = depth
        regressor = clf(max_depth = depth)
        # Calculate the training and testing scores
        sizes, train_scores, test_scores = curves(regressor, X, y, cv = cv, train_sizes = train_sizes)
        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis = 1)
        train_mean = np.mean(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)

        # Subplot the learning curve 
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha = 0.15, color = 'r')
        ax.fill_between(sizes, test_mean - test_std, test_mean + test_std, alpha = 0.15, color = 'g')

        # Labels
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])

    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize = 16, y = 1.03)
    fig.tight_layout()
    plt.show()
ModelLearning(features, prices,DecisionTreeRegressor)

#Complexity curve
#get the best max_depth for the model
def ModelComplexity(X, y, clf):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """
    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0],  test_size = 0.2, random_state = 0)
    # Vary the max_depth parameter from 1 to 10
    max_depth = np.arange(1,11)
    # Calculate the training and testing scores
    train_scores, test_scores = curvesvalidate(clf(), X, y, \
        param_name = "max_depth", param_range = max_depth, cv = cv, scoring = 'r2')
    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('Decision Tree Regressor Complexity Performance')
    plt.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')
    plt.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')
    plt.fill_between(max_depth, train_mean - train_std, train_mean + train_std, alpha = 0.15, color = 'r')
    plt.fill_between(max_depth, test_mean - test_std, test_mean + test_std, alpha = 0.15, color = 'g')
    # Visual aesthetics
    plt.legend(loc = 'lower right')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Score')
    plt.ylim([-0.05,1.05])
    plt.show()

ModelComplexity(X_train, y_train, DecisionTreeRegressor)


#Evaluating model performance,and fit it to reduce the error

def performance_metric(y_true, y_predict):
    """Calculates and returns the performance score between 
    true and predicted values based on the metric chosen. """
    #Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    #Return the score
    return score

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], test_size = 0.20, random_state = 0)
    #Create a decision tree regressor object
    regressor = DecisionTreeRegressor()
    #Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':np.arange(1,11)}
    #Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)
    #Create the grid search object
    grid = GridSearchCV(regressor, param_grid = params, scoring = scoring_fnc, cv = cv_sets)
    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)
    # Return the optimal model after fitting the data
    return grid.best_estimator_
# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)
print(reg)
# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

#make predictions
# Produce a matrix for client data
client_data = [[5, 34, 15], # Client 1
               [4, 55, 22]] # Client 2  

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client %d's home: $%.2f" %(i+1, price))

#Performs trials of fitting and predicting data.
def PredictTrials(X, y, fitter, data):
    # Store the predicted prices
    prices = []
    for k in range(10):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = k)
        # Fit the data
        reg = fitter(X_train, y_train)
        # Make a prediction
        pred = reg.predict([data[0]])[0]
        prices.append(pred)
        # Result
        print("Trial %d: $%.2f" %(k+1, pred))
    # Display price range
    print("\nRange in prices: $%.2f" %(max(prices) - min(prices)))

PredictTrials(features, prices, fit_model, client_data)









