import streamlit as st
import classification_dashboard
import comparison

# Front page. For now it just loads the classification dashboard, but in the future more things could be added. 

PAGES = {
    "Classification": classification_dashboard,
    "Direct Comparison": comparison
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()