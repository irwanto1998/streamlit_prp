# Import libraries
import streamlit as st
import pandas as pd
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import pickle


# Load the SVC model
svm = pickle.load(open('SVC_pricephonerange.pkl', 'rb')) 

# Load the mobile phone dataset
data = pd.read_excel('Price Range Phone Dataset.xlsx')

st.set_page_config(
    page_title="Phone App",
    page_icon="📱"
)

# Streamlit app
st.markdown("<h1 style='text-align: center;'>Price Range Phone App</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
    .rounded-button {
        background-color: purple;
        padding: 10px;
        border-radius: 15px; /* Sesuaikan dengan tingkat kebulatan yang diinginkan */
    }
</style>
""", unsafe_allow_html=True)

html_layout1 = """
<br>
<div class='rounded-button' style="background-color:purple ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Phone Specifications</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['SVM','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)

# Sidebar for user input
st.sidebar.header('DATA')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Dataset Price Range Phone merupakan kumpulan data yang berisi informasi tentang berbagai spesifikasi dari telepon seluler yang ada di pasaran. </p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#train test split
x = data.drop('price_range',axis=1)
y = data['price_range']
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('x_train')
    st.write(x_train.head())
    st.write(x_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('x_test')
    st.write(x_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

# Collect user input for mobile features
def user_report():
    battery_power = st.sidebar.slider('Battery Power (mAh)', 500, 5000, 2500)
    bluetooth = st.sidebar.radio('Bluetooth', ['Ada', 'Tidak Ada'])
    clock_speed = st.sidebar.slider('Clock Speed (GHz)', 0.5, 3.0, 1.0)
    dual_sim = st.sidebar.radio('Dual SIM', ['Ada', 'Tidak Ada'])
    fc = st.sidebar.slider('Front Camera (MP)', 0, 20, 5)
    four_g = st.sidebar.radio('4G', ['Ada', 'Tidak Ada'])
    internal_memory = st.sidebar.slider('Internal Memory (GB)', 8, 512, 64)
    m_dep = st.sidebar.slider('Mobile Depth (cm)', 0.1, 2.0, 0.5)
    mobile_wt = st.sidebar.slider('Weight of mobile phone (g)', 80, 300, 150)
    n_cores = st.sidebar.slider('Number of cores of processor', 1, 8, 4)
    pc = st.sidebar.slider('Primary Camera (MP)', 2, 40, 12)
    px_height = st.sidebar.slider('Pixel Resolution Height', 500, 2000, 1080)
    px_width = st.sidebar.slider('Pixel Resolution Width', 500, 2000, 1920)
    ram = st.sidebar.slider('RAM (GB)', 1, 12, 4)
    sc_h = st.sidebar.slider('Screen Height of mobile (cm)', 5, 20, 12)
    sc_w = st.sidebar.slider('Screen Width of mobile (cm)', 5, 20, 10)
    talk_time = st.sidebar.slider('Longest time that a single battery charge will last (hours)', 5, 50, 20)
    three_g = st.sidebar.radio('3G', ['Ada', 'Tidak Ada'])
    touch_screen = st.sidebar.radio('Touchscreen', ['Ada', 'Tidak Ada'])
    wifi = st.sidebar.radio('WiFi', ['Ada', 'Tidak Ada'])

    user_report_data = {
        'battery_power': [battery_power],
        'blue': [1 if bluetooth == 'Ada' else 0],  # Mengubah 'Ada' menjadi 1 dan 'Tidak Ada' menjadi 0
        'clock_speed': [clock_speed],
        'dual_sim': [1 if dual_sim == 'Ada' else 0],
        'fc': [fc],
        'four_g': [1 if four_g == 'Ada' else 0],
        'int_memory': [internal_memory],
        'm_dep': [m_dep],
        'mobile_wt': [mobile_wt],
        'n_cores': [n_cores],
        'pc': [pc],
        'px_height': [px_height],
        'px_width': [px_width],
        'ram': [ram],
        'sc_h': [sc_h],
        'sc_w': [sc_w],
        'talk_time': [talk_time],
        'three_g': [1 if three_g == 'Ada' else 0],
        'touch_screen': [1 if touch_screen == 'Ada' else 0],
        'wifi': [1 if wifi == 'Ada' else 0]
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data User
user_data = user_report()
st.subheader('Data Phone')
st.write(user_data)

def map_price_range(label):
    if label == 0:
        return 'Murah'
    elif label == 1:
        return 'Sedang'
    elif label == 2:
        return 'Mahal'
    elif label == 3:
        return 'Sangat Mahal'

prediction = svm.predict(user_data)[0]
predicted_price_range = map_price_range(prediction) 

#output
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Predicted Price Range:')
st.write(predicted_price_range)

# Menampilkan Grafik Distribusi Harga pada Sidebar
if st.checkbox('Distribusi Harga'):
    price_distribution = sns.countplot(x='price_range', data=data)
    
    # Mendapatkan gambar dari plot
    price_distribution_fig = price_distribution.get_figure()
    
    # Menampilkan gambar di Streamlit
    st.pyplot(price_distribution_fig)

# Menampilkan Ringkasan Statistik untuk Fitur Pilihan
selected_feature = st.selectbox('Pilih Fitur untuk Melihat Statistik', data.columns)
st.subheader(f'Ringkasan Statistik untuk {selected_feature}')
st.write(data[selected_feature].describe())

# Menampilkan Distribusi Fitur Pilihan
selected_feature_dist = st.selectbox('Pilih Fitur untuk Melihat Distribusi', data.columns)
st.subheader(f'Distribusi {selected_feature_dist}')

# Buat plot menggunakan Matplotlib
fig, ax = plt.subplots()
sns.histplot(data[selected_feature_dist], kde=True, ax=ax)

# Menampilkan gambar di Streamlit
st.pyplot(fig)

# Menampilkan Scatter Plot
scatter_x = st.selectbox('Pilih Fitur untuk Sumbu X', data.columns)
scatter_y = st.selectbox('Pilih Fitur untuk Sumbu Y', data.columns)
st.subheader(f'Scatter Plot antara {scatter_x} dan {scatter_y}')

# Buat plot menggunakan Matplotlib
fig, ax = plt.subplots()
sns.scatterplot(x=scatter_x, y=scatter_y, data=data, hue='price_range', ax=ax)

# Menampilkan gambar di Streamlit
st.pyplot(fig)









