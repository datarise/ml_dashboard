import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
#plt.style.use('ggplot')
from make_data import *
from classifications import *

random_state = 42

def app():
    """Runs the classification dashboard
    """    
    st.title('Sklearn Classification Playground')
    st.markdown('This is a dashbord for playing around with Sklearn classifiers. The main purpose, is to show how different classifiers perform and how their desicion bounderies look. It does not take crossvalidation into account. I have created this dashboard because I think it is fun to play around with Sklearn and Streamlit.')
    st.markdown('First we need to create some data to play around with. We need to select one of the three dataset creators from Sklearn "Make Blobs", "Make Moons" or "Make Circles". We can play around with the relevant paramters of the dataset after it is selected.')
    dataset = st.selectbox(
                        'Choose the dataset to create: ',
                        ('Make Blobs', 'Make Moons', 'Make Circles'))
    n_samples = st.slider('Choose the number of samles in the data: ', 0, 1000, 100, 50)
    noise = st.slider('Choose the noise level: ', 0.0, 1.0, 0.1, 0.05)
    
    X, y = make_classification_data(dataset, random_state, n_samples, noise)

    fig1 = plot_data(X, y, dataset, n_samples, noise)
    st.pyplot(fig1)

    st.markdown('We now have a dataset to play around with. We can now move on to select the wanted machine learning model we want to test.')
    st.markdown('We will start by selecting the ratio of the traning and test set. We will leave out cross-validation for now')

    test_size = st.slider('Choose the train/test raio: ', 0.0, 1.0, 0.1, 0.05)

    st.markdown('We now need to scale the data. We can chosse between the standard or min-max scaler method:')

    scaler = st.selectbox(
                    'Choose the scaler: ',
                    ('Standard Scaler', 'Min-Max Scaler'))

    st.markdown('We can now select the model we want to evaluate.')

    model = st.selectbox(
                'Choose the model: ',
                ('Logistic Regression', 'Desicion Tree', 'Random Forest', 'Support Vector Machine', 'Simple Neural Network'))

    st.markdown('The model parameters now needs to be choosen:')

    select_model(model, X, y, test_size, random_state, scaler)