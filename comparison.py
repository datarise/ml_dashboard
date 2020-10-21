import streamlit as st
from classifications import *
from make_data import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec

random_state = 42

def create_comparison_plot(select, models, X, y):

    fig1 = plt.figure(figsize=(15,10))
    plot_decision_regions(X, y, clf=models[0], legend=2)
    plt.title(select[0])
    st.pyplot(fig1)

    fig2 = plt.figure(figsize=(15,10))
    plot_decision_regions(X, y, clf=models[1], legend=2)
    plt.title(select[1])
    st.pyplot(fig2)

    fig3 = plt.figure(figsize=(15,10))
    plot_decision_regions(X, y, clf=models[2], legend=2)
    plt.title(select[2])
    st.pyplot(fig3)

    fig4 = plt.figure(figsize=(15,10))
    plot_decision_regions(X, y, clf=models[3], legend=2)
    plt.title(select[3])
    st.pyplot(fig4)
    #return fig

def select_model(model, X, y, random_state):

    if model == 'Logistic Regression':
        clf = train_LogisticRegression(X, y, random_state, True)
    elif model == 'Desicion Tree':
        clf = train_DecisionTreeClassifier(X, y, random_state, True)
    elif model == 'Random Forest':
        clf = train_RandomForestClassifier(X, y, random_state, True)
    elif model == 'Support Vector Machine':
        clf = train_SVC(X, y, random_state, True)
    elif model == 'Simple Neural Network':
        clf = train_MLP(X, y, random_state, True)
    
    return clf

def train_model(select, X, y, random_state):

    models = []
    for model in select:
        models.append(select_model(model, X, y, random_state))
    return models

def app():
    """Runs the comparison dashboard
    """    
    st.title('Sklearn Classification Playground')
    st.markdown('This is a dashbord for playing around with Sklearn classifiers. The main purpose, is to show how different classifiers perform and how their desicion bounderies look. It does not take crossvalidation into account. I have created this dashboard because I think it is fun to play around with Sklearn and Streamlit.')
    st.markdown('We first need to select the dataset we want to compare.')
    dataset = st.selectbox(
                        'Choose the dataset to create: ',
                        ('Make Blobs', 'Make Moons', 'Make Circles'))
    n_samples = st.slider('Choose the number of samles in the data: ', 0, 1000, 100, 50)
    noise = st.slider('Choose the noise level: ', 0.0, 1.0, 0.1, 0.05)
    
    X, y = make_classification_data(dataset, random_state, n_samples, noise)

    fig1 = plot_data(X, y, dataset, n_samples, noise)
    st.pyplot(fig1)

    st.markdown('We now need to select 4 classifiers to compare. The classifiers will just be trained with default parameters.')

    select = st.multiselect(
            'Choose the model: ',
            ('Logistic Regression', 'Desicion Tree', 'Random Forest', 'Support Vector Machine', 'Simple Neural Network'))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if len(select) != 4:
        st.write('Please select 4 classifers to continue.')
    else:
        models = train_model(select, X, y, random_state)
        create_comparison_plot(select, models, X, y)
        #st.pyplot(fig)
