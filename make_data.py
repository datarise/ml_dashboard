from sklearn.datasets import make_blobs, make_moons, make_circles
import streamlit as st

@st.cache(allow_output_mutation=True)
def make_classification_data(c_type, random_state, n_samples, noise):

    """Creates a dataset that can be used for ML testing. 

    Returns:
        X, y: X features, y target
    """    

    if c_type == 'Make Blobs':
        X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2, random_state=random_state, cluster_std=noise*10)
    elif c_type == 'Make Moons':
        X, y = make_moons(n_samples=n_samples, shuffle=True, noise=noise, random_state=random_state)
    elif c_type == 'Make Circles':
        X, y = make_circles(n_samples=n_samples, shuffle=True, noise=noise, random_state=random_state)
    
    return X, y






