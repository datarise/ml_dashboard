import streamlit as st 
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from classifications import * 

@st.cache()  
def load_data(csv):
    df = pd.read_csv(csv)
    return df 

#@st.cache() 
def profilling(df):
    pr = ProfileReport(df, explorative=True)
    st_profile_report(pr)


def create_model(df):

    X_select = st.multiselect(
        'Choose the features for the model: ',
        df.columns)
    if X_select:
        st.markdown("We now need to select the preprocessing steps that are needed for each column. Feature scaling is done later on. The dashboard will automaticly impute missing values.")
        preprocessing_feature = []
        preprocessing_feature_method = []
        for feature in X_select: 
            preprocess = st.multiselect(
                                    f"Please choose the preprocessing steps that are needed for {feature}",
                                    ('Nothing', 'Label Encoding', 'One-Hot Encoding'))
            if preprocess:
                preprocessing_feature.append(feature)
                preprocessing_feature_method.append(preprocess[0])
            
            
        y_select = st.selectbox(
        'Choose the target variable for den model: ',
        X_select)

        preprocessing_feature.append(y_select)
        preprocessing_feature_method.append('Target')

        X, y = preprocessing(preprocessing_feature, preprocessing_feature_method, df)

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

        select_model(model, X, y, test_size, 42, scaler)
            

def preprocessing(preprocessing_feature, preprocessing_feature_method, df):
    df_X = pd.DataFrame()
    for feature, process in zip(preprocessing_feature, preprocessing_feature_method):
        if process == "Label Encoding":
            le = LabelEncoder()
            df_X[feature] = le.fit_transform(df[feature])
        elif process == "One-Hot Encoding":
            df_temp = pd.get_dummies(df[feature], drop_first=True)
            df_X = pd.concat([df_X, df_temp], axis=1)
        else: 
            df_X[feature] = df[feature].values

    imp = IterativeImputer(max_iter=10, verbose=0)
    imp.fit(df_X)
    imputed_df = imp.transform(df_X)
    df_X = pd.DataFrame(imputed_df, columns=df_X.columns)

    y = df_X[preprocessing_feature[-1:][0]].values.astype(np.integer)
    X = df_X.drop(preprocessing_feature[-1:][0], axis=1).values

    return X, y



def app():
    st.title("Sklearn Playground")
    st.markdown("Here can you play around classifing your own data. Start by uploading your data as a csv.")
    csv = st.file_uploader("Upload your data set as csv", type="csv")

    if csv:
        df = load_data(csv)
        st.markdown("An overview of the data.")
        st.write(df)
        st.markdown("A exploratory data analysis is displayed below if the buttons is clicked. It's made with Pandas Profiling and can take a while to run. You can therefor turn it of when you are done.")
        show_EDA = st.radio("Show EDA", ("No", "Yes"))
        if show_EDA == "Yes":
            profilling(df)
        

        st.markdown("We now need to select the features and target variable that we want to use.")
        
        selected_variables = create_model(df)





