
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os
import io
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
def write_arabic(text):
    return f'<h2 dir="rtl">{text}</h2>'
def num_numerica_and_categorical(data):
    numeric_columns = data.select_dtypes(include=['int', 'float'])
    categorical_columns = data.select_dtypes(include=['object'])
    return numeric_columns,categorical_columns
def read_file(uploaded_file):
     return pd.read_csv(uploaded_file)
def save_file(data,file_path):
    data.to_csv(file_path, index=False)
def save_array(x,y):
     with open("data.pkl", "wb") as file:
        pickle.dump(x, file)
        pickle.dump(y, file)
def load_array():
     with open("data.pkl", "rb") as file:
        loaded_x = pickle.load(file)
        loaded_y = pickle.load(file)
     return loaded_x,loaded_y
     
    
def auto_encoder_detection(data, threshold=0.5):
                    categorical_columns = data.select_dtypes(include=['object'])
                    columns_to_encode = []

                    for column in categorical_columns.columns:
                        unique_ratio = len(data[column].unique()) / len(data)
                        if unique_ratio < threshold:
                            columns_to_encode.append(column)
                    return columns_to_encode

def refresh():
     st.button("Refresh")

def main():
   
    y=None
    
   
    
    st.set_option('deprecation.showPyplotGlobalUse', True)
    st.title("AutoMLTool - Your AutoML Assistant")
    
    st.write(write_arabic("""هذا التطبيق البسيط يقوم بتحليل وتدريب البيانات حيث يقوم بتحميل الملف وعرض بعض الإحصاءات مع عمل تحليل استكشافي ومعالجة البيانات و  عمل visulization من خلال رسم رسوم بيانية مختلفة كالهيستوجرام والبار شارت والمصفوفة الحرارية للعلاقة بين المتغيرات ثم تدريب نماذج Machine Learning مختلفة مع حساب مقاييس الأداء ورسم confusion matrix  ."""), unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
       
        file_path = os.path.join(uploaded_file.name)
        if  os.path.exists(uploaded_file.name):
           pass
        else:
          
           save_file(data,file_path)  
    
     
       
       
        numeric_columns,categorical_columns=  num_numerica_and_categorical(data)
        targetFeature = st.sidebar.selectbox("select target Feature", data.columns)
        
        if targetFeature:
           label_encoder = LabelEncoder()
           type_target=data.eval(targetFeature).value_counts()
           if len(type_target) > 10 and  data.eval(targetFeature).dtypes in [np.float64,np.int64]:
                target_type="Regression"
           elif  len(type_target) == 2:
                target_type="Binary classification"
           else:
                 target_type="multi classification"
        if (len(type_target)<10):
                y= label_encoder.fit_transform(data[targetFeature])
        else:
      

            y=data[targetFeature]


                 
        
               
        desc={ "Description":["Target Class variable","Target type","Original data shape","Numeric features","Categorical features"],
                        "value":[targetFeature,target_type,data.shape,numeric_columns.shape[1],categorical_columns.shape[1]]}
        

        
        

        st.markdown("""<hr style="height:10px;border:none;color:#efefef;background-color:#efefef;" /> """, unsafe_allow_html=True)
        loadData=st.checkbox('Exploratory data analysis (EDA)')
        if loadData:
            nameTabs=["show  data", "Describe Data", "columns names","data info",'general information','check value is numll']
            tab1, tab2, tab3 ,tab4,tab5,tab6= st.tabs(nameTabs)
            with tab1:
                st.header("See a set of data")
                selectShowColumns = st.selectbox("",['First five',  'Last five','Random sample'])
                if selectShowColumns=="First five":
                    st.write(data.head())
                elif selectShowColumns=="Last five":
                    st.write(data.tail())
                elif selectShowColumns=="Random sample":
                    st.write(data.sample(5))

            with tab2:
                st.header("Describe Data")
                st.write(data.describe())

            with tab3:
                st.header("Columns names ")
                st.write(data.columns)
            with tab4:
                st.header("DataFrame information ")
                buffer = io.StringIO()
                data.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s[s.index("#"):])
            with tab5:
               st.header("See general information about the data: ")
               st.write(pd.DataFrame(desc))
            with tab6:
                st.header("check value is null ")
                st.table(data.isnull().sum())
            
            
           
            
                
          
         
                
            
            
                
          
              
         
            
           
                
             
        st.markdown("""<hr style="height:10px;border:none;color:#efefef;background-color:#efefef;" /> """, unsafe_allow_html=True)        
        #visulization
        visulization=st.checkbox('visulization data')
        if visulization:
             nameTabs=["A bar chart ", "histogram", "draw two column","heatmap"]
             tab1, tab2, tab3 ,tab4= st.tabs(nameTabs)

             with tab1:
                
                  st.header("A bar chart  ")
                  
                 
                  columns =categorical_columns.columns

                  for column in columns:
                       
                        value_counts = data[column].value_counts()
                     
                        st.title(f"columns { column}")
                        sns.set(style="whitegrid")
                        plt.figure(figsize=(8, 6))
                        sns.countplot(x=value_counts, data=data)
                        
                        plt.title(f"value counts column { column}")
                        plt.xlabel(f"{column}")
                        plt.ylabel('count')
                        st.pyplot(plt)
                        st.divider()
             with tab2:
                
                  features = st.multiselect("Select feature", numeric_columns.columns)
                  if features:
                    data[features].hist(figsize=(10,5))
                    st.pyplot(plt)
                 
                 
             with tab3:
                    st.title("Two Select Boxes Side by Side")
                  
                 
                    left_column, center_column ,right_column= st.columns(3)

               
                    with left_column:
                        option1 = st.selectbox("اختر الخيار الأول", numeric_columns.columns)

                
                    with center_column:
                        option2 = st.selectbox("اختر الخيار الثاني",numeric_columns.columns)
                    with right_column:
                        hue = st.selectbox("",data.columns)
                    
                    plt.figure(figsize=(8, 6))
                    if st.checkbox("True hue"):
                        sns.jointplot(x=data[option1], y=data[option2],data=data,kind='scatter',hue=hue)
                    else:
                        sns.jointplot(x=data[option1], y=data[option2],data=data,kind='scatter')
                    plt.title(f"value counts column ")
                    plt.xlabel(f"s")
                    plt.ylabel('count')
                    st.pyplot(plt)
                
                    st.divider()
             with tab4:
                    st.header(" heatmap ")
                   
                    plt.figure(figsize=(15,15))
                    sns.heatmap(numeric_columns.corr(),annot=True)
                    st.pyplot(plt)

                   
                   

        st.markdown("""<hr style="height:10px;border:none;color:#efefef;background-color:#efefef;" /> """, unsafe_allow_html=True)
        preprocessing=st.checkbox('preprocessing')
        if preprocessing:
             refresh()
             nameTabs=["instruction", "Drop columns", "Full null","OneHotEncoder","StandardScaler"]
             tab1, tab2, tab3 ,tab4,tab5= st.tabs(nameTabs)
             with tab1:
                    st.write(write_arabic("اختر target column من القائمة الجانبية"), unsafe_allow_html=True)
                    st.write(write_arabic("اتبع الخطوات التالية: "), unsafe_allow_html=True)
                    st.write(write_arabic("1- حذف الأعمدة غير الضرورية"), unsafe_allow_html=True)
                    st.write(write_arabic("2- ملء القيم الفارغة"), unsafe_allow_html=True)
                    st.write(write_arabic("3- OneHotEncoder"), unsafe_allow_html=True)
                    st.write(write_arabic("4- StandardScaler"), unsafe_allow_html=True)
                    st.write(write_arabic("بعد إجراء هذه العمليات، يجب تحديث الملف من جديد حيث ستتم حفظ نسخة جديدة من الملف في نفس المسار"), unsafe_allow_html=True)

                        
             with tab2:
                data=read_file(uploaded_file.name)
                st.write(uploaded_file.name)
                numeric_columns,categorical_columns=  num_numerica_and_categorical(data)
                st.header("Select the columns you want to delete and target columns")
                columns_delete = st.multiselect("Select column", data.columns)
                
                if st.button("Delete"):

                    if columns_delete is not None:
                       
                        data =data = data.drop(columns=columns_delete, axis=1)
                        st.write(data)
                        st.success("تم حذف الأعمدة بنجاح.")
                st.write(data)

                save_file(data,uploaded_file.name)
             with tab3:
                    if st.button("full is null "):
                        data=read_file(uploaded_file.name)
                        numeric_columns,categorical_columns=  num_numerica_and_categorical(data)
                        data[numeric_columns.columns] = data[numeric_columns.columns].fillna(data[numeric_columns.columns].mean())
                        for column in categorical_columns.columns:
                            mode_value = data[column].mode().values[0]
                            data[column].fillna(mode_value, inplace=True)
                        st.write(data)
                        save_file(data,uploaded_file.name)
             with tab4:
         
                data=read_file(uploaded_file.name)
                if st.button("OneHotEncoder  "):
                        st.write("auto")
                        columns_to_encode = auto_encoder_detection(data, threshold=0.5)
                        st.write(columns_to_encode)
                        encoder = OneHotEncoder(sparse=False, drop='first') 
                        encoded_data = encoder.fit_transform(data[columns_to_encode])
                        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns_to_encode))
                        data = data.drop(columns=columns_to_encode)
                        data = pd.concat([data, encoded_df], axis=1)
                   
                st.write(data)
                save_file(data,uploaded_file.name)
             with tab5:
                data=read_file(uploaded_file.name)
                if st.button("StandardScaler"):
                    x = data
                    scl = MinMaxScaler()
                    x_scaled = scl.fit_transform(x)
                    st.header("x scaled ")
                    st.write(x_scaled)
                    st.write("*"*50)
                    st.header("Y ")
                    st.write(y)
                    save_array(x,y)
                

    






        st.markdown("""<hr style="height:10px;border:none;color:#efefef;background-color:#efefef;" /> """, unsafe_allow_html=True)
        try :
                train_model=st.checkbox('train model')


                if train_model:
                    x,y=load_array()
                    
                    select_test_size= st.slider("Select a threshold", min_value=0.1, max_value=0.8, value=0.2, step=0.1)
                    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=select_test_size, random_state=42)
                    if target_type in ["Binary classification",'multi classification']:
                        model_select=['K-Nearest Neighbors','DecisionTreeClassifier','RandomForestClassifier']
                    else:
                        
                        model_select=["LinearRegression","PolynomialFeatures","Lasso Regression"]
                    select_model = st.selectbox("select model",model_select) 
                    if select_model=="LinearRegression":
                        model=LinearRegression()
                    
                    elif select_model=="PolynomialFeatures":
                        select_degree= st.slider("Select degree", min_value=1, max_value=8, value=2, step=1)
                        poly_features = PolynomialFeatures(degree=select_degree)
                        model=LinearRegression()
                        X_train = poly_features.fit_transform(X_train)
                        X_test = poly_features.transform(X_test)
                    elif select_model=="Lasso Regression":
                        select_alpha= st.slider("Select a alpha", min_value=0.1, max_value=0.8, value=0.2, step=0.1)
                        model = Lasso(alpha=select_alpha) 
                    elif select_model=="K-Nearest Neighbors":
                        number = st.number_input("Enter  a number n neighbors ", min_value=1, max_value=20, value=3, step=2)

                        X_train = np.array(X_train)
                        X_test = np.array(X_test)
                    
                        model=KNeighborsClassifier(n_neighbors=number)
                        
                    elif select_model=="DecisionTreeClassifier":
                        max_depth = st.number_input("Enter  a number n neighbors ", min_value=1, max_value=20, value=3, step=1)
                        
                        model= DecisionTreeClassifier(max_depth=max_depth)
                    elif select_model=="RandomForestClassifier":
                        n_estimators = st.number_input("Enter  a number n neighbors ", min_value=1, max_value=200, value=25, step=1)
                        
                        model= RandomForestClassifier(n_estimators=n_estimators)
                        
                    model.fit(X_train, y_train)
                    preds=model.predict(X_train)
                    if (len(type_target)<10):
                        st.write("Accuracy Train : ",accuracy_score(y_train,preds))
                        test_predict=model.predict(X_test)
                        st.write("Accuracy test : ", accuracy_score(y_test,test_predict))
                        cm = confusion_matrix(y_test, test_predict)
                        plt.figure(figsize=(8, 6))
                        sns.set(font_scale=1.2)  # Adjust font size for better visualization
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 12})

                        # Add labels and title
                        plt.xlabel("Predicted Labels")
                        plt.ylabel("True Labels")
                        plt.title("Confusion Matrix")
                        st.pyplot(plt)
                    else:
                        st.write("Accuracy Train : ",r2_score(y_train,preds))
                        test_predict=model.predict(X_test)
                        st.write("Accuracy Test : ",r2_score(y_test,test_predict))
                    
                    if st.button("delte temp file"):
                        os.remove(uploaded_file.name)
                        os.remove("data.pkl")
                        st.experimental_rerun()
        except FileNotFoundError:
          
            st.write("The specified file does not exist.")
            st.write("يجب معالجة ان يتم معالجة البيانات")
        except Exception as e:
          
            st.write(f"An error occurred: {e}")
    else:
        st.write("*"*50)
        st.write(write_arabic("قم باختيار الملف من القائمة الجانبية"), unsafe_allow_html=True)

            





























    

    
if __name__ == "__main__":
    main()
