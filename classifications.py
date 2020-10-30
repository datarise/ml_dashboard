from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, plot_roc_curve
import streamlit as st
import numpy as np
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

def plot_data(X, y, dataset, n_samples, noise):
    """Plots 

    Args:
        X (numpy array): features
        y (numpy array): target
        classifier (sklearn classifier): ML Model
        n_samples (int): number of samples
        noise (float): noise for the data

    Returns:
        matplotlib figure: figure of data
    """    
    fig = plt.figure(figsize=(15,10))
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y)
    #plt.set_xlabel("X_axis_title")
    #plt.set_ylabel("Y_axis_title")
    plt.title(f'{dataset} data with {n_samples} samples and a noise level of {noise}')
    return fig


def select_model(model, X, y, test_size, random_state, scaler):
    """Selects a ML model to train, creates a test and train dataset  

    Args:
        model (string): String name for the classifier to be trained
        X (numpy array): Feature for the model
        y (numpy array): Target for the model
        test_size (float): train/test ratio between 0-1
        random_state (int): Seed 
        scaler (String): String name for the scaler
    """    

    X_train, X_test, y_train, y_test, X= split_and_scale(scaler, test_size, random_state, X, y)

    if model == 'Logistic Regression':
        clf = train_LogisticRegression(X_train, y_train, random_state, False)
    elif model == 'Desicion Tree':
        clf = train_DecisionTreeClassifier(X_train, y_train, random_state, False)
    elif model == 'Random Forest':
        clf = train_RandomForestClassifier(X_train, y_train, random_state, False)
    elif model == 'Support Vector Machine':
        clf = train_SVC(X_train, y_train, random_state, False)
    elif model == 'Simple Neural Network':
        clf = train_MLP(X_train, y_train, random_state, False)
    
    evaluation(clf, X_train, X_test, y_train, y_test, X, y)

def split_and_scale(scaler, test_size, random_state, X, y):
    """Creates train and test sets and scales them

    Args:
        scaler (String): String name for the scaler
        test_size (float): train/test ratio between 0-1
        random_state (int): Seed 
        X (numpy array): Feature for the model
        y (numpy array): Target for the model

    Returns:
        numpy arrays: train and test data
    """    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if scaler == 'Standard Scaler':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X = scaler.transform(X)
    else:
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X = scaler.transform(X)

    return X_train, X_test, y_train, y_test, X

def train_LogisticRegression(X, y, random_state, compare):
    """Fits a logistic regression and displays it's parameters

    Args:
        X (numpy array): Feature for the model
        y (numpy array): Target for the model
        random_state (int): Seed 
        compare (bool): if the classifiers is to be used for comparison

    Returns:
        sklearn classifier: logistic regression
    """    
    if compare:
        clf = LogisticRegression(random_state=random_state)
        clf.fit(X, y)
    else:
        penalty = st.selectbox(
                'Choose the penalty: ',
                ("l1", "l2", "elasticnet", "none"))
        
        C = st.slider('Choose the C Value: ', 0.01, 100.0, 0.1, 0.05)

        l1_ratio = st.slider('Choose the L1/L2 ratio: ', 0.0, 1.0, 0.1, 0.05)

        fit_intercept = st.selectbox(
            'Choose if the intercept should be fitted: ',
            ("True", "False"))

        clf = LogisticRegression(penalty=penalty, solver='saga', C=C, random_state=random_state, fit_intercept=bool(fit_intercept), n_jobs=-1, l1_ratio=l1_ratio)
        clf.fit(X, y)

    return clf

def train_DecisionTreeClassifier(X, y, random_state, compare):
    """Fits a decision tree and displays it's parameters

    Args:
        X (numpy array): Feature for the model
        y (numpy array): Target for the model
        random_state (int): Seed 
        compare (bool): if the classifiers is to be used for comparison

    Returns:
        sklearn classifier: decision tree
    """ 
    if compare:
        clf = DecisionTreeClassifier(random_state=random_state, max_depth=6)
        clf.fit(X, y)
    else:
        criterion = st.selectbox(
                'Choose the criterion: ',
                ('gini', 'entropy'))
        
        max_depth = st.slider('Choose Max Depth: ', 1, 100, 10, 1)

        min_samples_split = st.slider('Choose minimum sample split: ', 2, 25, 2, 1)

        min_samples_leaf = st.slider('Choose minimum sample leaf: ', 1, 25, 1, 1)

        ccp_alpha = st.slider('Choose the pruning ratio: ', 0.0, 0.2, 0.0, 0.005)

        clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha)

        clf.fit(X, y)

    return clf

def train_RandomForestClassifier(X, y, random_state, compare):
    """Fits a random forest and displays it's parameters

    Args:
        X (numpy array): Feature for the model
        y (numpy array): Target for the model
        random_state (int): Seed 
        compare (bool): if the classifiers is to be used for comparison

    Returns:
        sklearn classifier: random forest
    """    
    if compare:
        clf = RandomForestClassifier(random_state=random_state, n_estimators=5, max_depth=6)
        clf.fit(X, y)
    else:
        criterion = st.selectbox(
                'Choose the criterion: ',
                ('gini', 'entropy'))
        
        max_depth = st.slider('Choose Max Depth: ', 1, 100, 10, 1)

        min_samples_split = st.slider('Choose minimum sample split: ', 1, 25, 2, 1)

        min_samples_leaf = st.slider('Choose minimum sample leaf: ', 1, 25, 1, 1)

        n_estimators = st.slider('Choose the number of estimators: ', 1, 100, 10, 1)

        clf = RandomForestClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, n_jobs=-1)

        clf.fit(X, y)

    return clf

def train_SVC(X, y, random_state, compare):
    """Fits a support vector machine and displays it's parameters

    Args:
        X (numpy array): Feature for the model
        y (numpy array): Target for the model
        random_state (int): Seed 
        compare (bool): if the classifiers is to be used for comparison

    Returns:
        sklearn classifier: support vector machine
    """    
    if compare:
        clf = SVC(random_state=random_state)
        clf.fit(X, y)
    else:
        kernel = st.selectbox(
                'Choose the kernel: ',
                ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'))
        
        C = st.slider('Choose Regularization Parameter C: ', 0.1, 100.0, 1.0, 0.25)

        degree = st.slider('Choose degree for polynomial kernel: ', 1, 25, 3, 1)

        clf = SVC(random_state=random_state, kernel=kernel, C=C, degree=degree)

        clf.fit(X, y)

    return clf

def train_MLP(X, y, random_state, compare):
    """Fits a neural network and displays it's parameters

    Args:
        X (numpy array): Feature for the model
        y (numpy array): Target for the model
        random_state (int): Seed 
        compare (bool): if the classifiers is to be used for comparison

    Returns:
        sklearn classifier: neural network
    """    
    if compare:
        clf = MLPClassifier(random_state=random_state, hidden_layer_sizes=(20,20), activation="relu")
        clf.fit(X, y)
    else:
        activation = st.selectbox(
                'Choose the activation function: ',
                ('identity', 'logistic', 'tanh', 'relu'))

        # learning_rate = st.selectbox(
        #     'Choose the learning rate: ',
        #     (0.001, 0.001, 0.01, 0.1, 1))
        
        h1 = st.slider('Choose the number of hidden unit in layer 1: ', 1, 100, 10, 1)
        h2 = st.slider('Choose the number of hidden unit in layer 2: ', 1, 100, 10, 1)

        clf = MLPClassifier(random_state=random_state, hidden_layer_sizes=(h1, h2), activation=activation, max_iter=1000)

        clf.fit(X, y)

    return clf


def plot_desicion_boundery_n_roc(X, y, clf, X_test, y_test):
    """Plots the desicioon boundery and ROC-curve

    Args:
        X (numpy array): features
        y (numpy array): target
        clf (sklearn classifier): ML model
        X_test (numpy array): test features
        y_test (numpy array): test target
    """    
    if X.shape[1] == 2:
        fig1 = plt.figure(figsize=(15,10))
        plot_decision_regions(X, y, clf=clf, legend=2)
        # Adding axes annotations
        plt.title('Decision Boundery')
        st.pyplot(fig1)

    if len(np.unique(y)) == 2:
        st.markdown('The ROC-curve for the test data is displayed below:')
        fig2 = plt.figure(figsize=(15,10))
        ax = fig2.add_subplot(111)
        plot_roc_curve(clf, X_test, y_test, ax=ax) 
        st.pyplot(fig2)



def evaluation(clf, X_train, X_test, y_train, y_test, X, y):
    """Evaluates the model with a classification report

    Args:
        clf (sklearn classifier): ML model
        X_train (numpy array): train features
        X_test (numpy array): test features
        y_train (numpy array): train target
        y_test (numpy array): test target
        X (numpy array): features
        y (numpy array): target
    """    
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    target_names = []
    for i in range(len(np.unique(y_train))):
        target_names.append(f'Class {i}')

    st.markdown('The classification report for the training data is displayed below:')
    st.text(classification_report(y_train, y_train_pred, target_names=target_names))

    st.markdown('The classification report for the test data is displayed below:')
    st.text(classification_report(y_test, y_test_pred, target_names=target_names))

    plot_desicion_boundery_n_roc(X, y, clf, X_train, y_train)