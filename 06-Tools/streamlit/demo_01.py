import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# 1. st.write()
st.write("1. st.write()")
st.write(pd.DataFrame({
    'col1': [1, 2, 3, 4],
    'col2': [10, 20, 30, 40]})
)

# 2. st.line_chart()
st.write("2. st.line_chart()")
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)
st.line_chart(chart_data)

# 3. st.map()
st.write("3. st.map()")
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon']
)
st.map(map_data)

# 4. st.slinder()
st.write('4. st.slider()')
x = st.slider('x')
st.write(x, 'squared is', x*x)

# 5. st.text_input()

# 6. st.checkbox()

# 7. st.selectbox()

# 8. st.sidebar()

# 9. st.radio()

# 10. st.progress()

# 11. st.file_uploader()



