import streamlit as st
import requests
import pandas as pd
import pprint as pprint
# Get data from Flask API

def generate_metrics():
    response = requests.get("https://jsonplaceholder.typicode.com/todos")
    data = response.json()

    # Create a dataframe
    df = pd.DataFrame({
        'id': el['id'],
        'title': el['title']
    } for el in data)

    # Display in Streamlit
    st.title("Data from Flask API")
    st.write(df)
    st.bar_chart(df.set_index('id'))


def page_2():
    st.title("page 2")

def about_me():
    st.title("Hi, I'm Franco Gutierrez a Software Engineer")
    st.header("I am based in Ireland")
    st.header("And this is streamlit", divider="gray")
    st.header("Using python", divider=True)

def generate_navigation():
    pages = {
    "Your account": [
        st.Page(generate_metrics, title="Create your account"),
        st.Page(page_2)
       
    ],
    "Resources": [
        st.Page(about_me)
    ]
}

    pg = st.navigation(pages)
    pg.run()

if __name__ == '__main__':
   generate_navigation()