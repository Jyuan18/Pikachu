import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# # 1. st.write()
# st.write("1. st.write()")
# st.write(pd.DataFrame({
#     'col1': [1, 2, 3, 4],
#     'col2': [10, 20, 30, 40]})
# )

# # 2. st.line_chart()
# st.write("2. st.line_chart()")
# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=['a', 'b', 'c']
# )
# st.line_chart(chart_data)

# # 3. st.map()
# st.write("3. st.map()")
# map_data = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#     columns=['lat', 'lon']
# )
# st.map(map_data)

# # 4. st.slinder()
# st.write('4. st.slider()')
# x = st.slider('x')
# st.write(x, 'squared is', x*x)

# 5. st.text_input()

# 6. st.checkbox()

# 7. st.selectbox()

# 8. st.sidebar()

# 9. st.radio()

# 10. st.progress()

# 11. st.file_uploader()



st.text('Fixed width text')
st.markdown('_Markdown_') # see #*
st.caption('Balloons. Hundreds of them...')
st.latex(r''' e^{i\pi} + 1 = 0 ''')
st.write('Most objects') # df, err, func, keras!
st.write(['st', 'is <', 3]) # see *
st.title('My title')
st.header('My header')
st.subheader('My sub')
st.code(
    '''
    for i in range(8):
        foo()
        return sss
    '''
)

# * optional kwarg unsafe_allow_html = True

# Just add it after st.sidebar:
a = st.sidebar.radio('Choose:',[1,2])

data = {'a':1}

st.button('Hit me')
# st.data_editor('Edit data', data)
st.checkbox('Check me out')
st.radio('Pick one:', ['nose','ear'])
st.selectbox('Select', [1,2,3])
st.multiselect('Multiselect', [1,2,3])
st.slider('Slide me', min_value=0, max_value=10)
st.select_slider('Slide to select', options=[1,'2'])
st.text_input('Enter some text')
st.number_input('Enter a number')
st.text_area('Area for textual entry')
st.date_input('Date input')
st.time_input('Time entry')
st.file_uploader('File uploader')
st.download_button('On the dl', data)
st.camera_input("一二三,茄子!")
st.color_picker('Pick a color')