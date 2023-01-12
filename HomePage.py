import streamlit as st
import pandas as pd 
import numpy as np 

st.set_page_config(
    page_title="Name Generator",
    page_icon="🥼"
)


st.title('Medical Brand Name Generator')
st.sidebar.success("Select a page above.")

if 'molecule' not in st.session_state:
    st.session_state['molecule'] = ''

molecule_Name = st.text_input("Enter the Molecule Name",st.session_state['molecule'])


if 'country' not in st.session_state:
    st.session_state['country'] = ''

country_Name = st.text_input("Enter the Country Name",st.session_state['country'])
 



submit = st.button('submit')
if submit: 
    st.session_state['molecule'] = molecule_Name
    st.session_state['country'] = country_Name
    
    st.write("You have entered: ",country_Name, molecule_Name)

