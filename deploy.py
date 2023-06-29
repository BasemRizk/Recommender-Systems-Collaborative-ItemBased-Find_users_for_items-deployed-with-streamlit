from recommendation_engine import *
import streamlit as st
import pandas as pd
import pickle
with open('my_list.pkl', 'rb') as f:
    items = pickle.load(f)
with open('users.pkl', 'rb') as f:
    users = pickle.load(f)
# Create a selectbox for choosing a fruit
selected_item = st.selectbox('Select an item code:', items)

# Print the selected fruit

model_names = ['More like this item', 'Favourite Clients for this item', 'More like this item for specific user']

selected_model = st.radio('Select a model:', [None] + model_names)


if selected_item is not None:
    if selected_model==model_names[0]:
        st.title('Results')
        x1=get_scores(selected_item).reset_index()
        x1.index+=1
        x1[['StockCode','Desc','UnitPrice']]
    elif selected_model==model_names[1]:
        st.title('Results')
        x2 = fav_clients(selected_item).reset_index()
        x2.index+=1
        x2=x2[['CustomerID']]
        st.write(x2)
    elif selected_model==model_names[2]:
        selected_user = st.selectbox('Select a user ID:',[None]+ users)
        st.write('You selected:', selected_user)
        if selected_user is not None:
            st.title('Results')
            x3=(colab_filter(selected_user,selected_item).reset_index())
            x3.index+=1
            x3=x3[['StockCode','Desc','UnitPrice']]
            st.write(x3)