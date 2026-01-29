import streamlit as st
import pandas as pd
from numpy.random import default_rng as rng

st.title("Streamlit Application")
st.header("Welcome to the Streamlit App")
st.subheader("This is a subheader")
st.text("This is a simple Streamlit application demonstrating headers and text.")
st.write("You can use `st.write()` to display various types of content.")
st.markdown("### This is a markdown header")
st.caption("This is a caption text.")

name = st.text_input("Enter your name:")
st.write(f"Hello, {name}!")

age = st.number_input("Enter your age:", 18,120,20)
st.write(f"You are {age} years old.")

gender = st.radio("Select your Gender:", ("Male", "Female", "Other"))
DOB = st.date_input("Select your birth date:")
Hobbies = st.multiselect("Select your hobbies:", ["Reading", "Traveling", "Gaming", "Cooking"])
rate = st.slider("Rate your experience:", 1, 10, 5)


Data = pd.DataFrame(
    {
        "Name": [name],
        "Gender": [gender],
        "Age": [age],
        "DOB": [DOB],
        "Hobbies": [", ".join(Hobbies)],
        "Experience Rating": [rate],
    },
    index=["Name", "Gender", "Age", "DOB", "Hobbies", "Experience Rating"],
)
st.table(Data)

df = pd.DataFrame(rng(0).standard_normal((20, 3)), columns=["a", "b", "c"])

st.line_chart(df)
st.scatter_chart(df)
st.bar_chart(df)

scatter_data = pd.DataFrame({
    "x" : [5,7,8,7,2,17,2,9,4,11,12,9,6],
    "y" : [99,86,87,88,100,86,103,87,94,78,77,85,86],
    "category" : [1,2,2,1,1,2,1,2,1,2,2,1,2]    
})
st.scatter_chart(scatter_data)

# st.vega_lite_chart(
#     scatter_data, {
#         "mark": "point",
#         "encoding": {
#             "x": {"field": "x", "type": "quantitative"},
#             "y": {"field": "y", "type": "quantitative"},
#             "color": {"field": "category", "type": "nominal"},
#         },
#     },
# )

st.success('This is a success message!', icon="✅")
st.warning('This is a warning', icon="⚠️")
st.error('This is an error', icon="🚨")
st.balloons()