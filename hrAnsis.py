import streamlit as st
import streamlit.components.v1 as components
from streamlit_player import st_player
import sweetviz as sv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from PIL import Image

le = LabelEncoder()

st.sidebar.header("""
                    Human Resource Analysis
                    by NI.S
                    """)

menu = st.sidebar.selectbox('Menu',['Home','T&C','Human Resource EDA','Human Resource Analysis Dashboard','Human Resource Prediction'])

if menu == 'Home':
  st.write("""
    # Welcome!
    ## Human Resource Analysis
    This app gathers the data related to employees in a company, and we'll see the relationship of the features and do predictions.
    Training data used is from [here](https://www.kaggle.com/colara/human-resource)

    ## Explanation Video
    """)
  # Embed a youtube video, hint from https://discuss.streamlit.io/t/streamlit-player/3169
  st_player("https://youtu.be/zHB1vsB2Fz0")
  st.write("## Get to know me!")
  image = Image.open('izzati.jpg')
  st.image(image, caption='The girl is me; Nurul Izzati binti Suhardi. Image was taken during The 4th Creative Robot Contest for Decommissioning 2019 in Fukushima, Japan.')
  st.write("""
  If you wish to connect with me:
    - [LinkedIn](https://www.linkedin.com/in/nurul-izzati-suhardi/)
    - [GitHub](https://github.com/NIS-1213)
    - Email: nurulizzati120198@gmail.com

  Thank you for visiting the app!
  """)

elif menu == 'T&C':
  st.write('Before you continue, please read the [terms and conditions](https://www.gnu.org/licenses/gpl-3.0.en.html)')
  show = st.checkbox('I agree the terms and conditions')

elif menu == 'Human Resource Analysis Dashboard':

  st.write("""
  # Human Resource Dashboard
  via [Tableau](https://public.tableau.com/views/HumanResourceDashboardAAxUTM/HumanResourceAnalysisDashboard?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link)
  """)
  
  def dashboard():

    html_temp = """
      <div class='tableauPlaceholder' id='viz1646892856779' style='position: relative'><object class='tableauViz'  
      style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> 
      <param name='embed_code_version' value='3' /> 
      <param name='site_root' value='' /><param name='name' value='HumanResourceDashboardAAxUTM&#47;HumanResourceAnalysisDashboard' />
      <param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' />
      <param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                
      <script type='text/javascript'>                    var divElement = document.getElementById('viz1646892856779');                    
      var vizElement = divElement.getElementsByTagName('object')[0];                    
      if ( divElement.offsetWidth > 800 ) { vizElement.style.minWidth='360px';vizElement.style.maxWidth='1920px';vizElement.style.width='100%';vizElement.style.minHeight='667px';vizElement.style.maxHeight='1107px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} 
      else if ( divElement.offsetWidth > 500 ) { vizElement.style.minWidth='360px';vizElement.style.maxWidth='1920px';vizElement.style.width='100%';vizElement.style.minHeight='667px';vizElement.style.maxHeight='1107px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} 
      else { vizElement.style.width='100%';vizElement.style.height='2127px';}                     
      var scriptElement = document.createElement('script');                    
      scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    
      vizElement.parentNode.insertBefore(scriptElement, vizElement);                
      </script>
      """
    components.html(html_temp, width = 1080, height = 800, scrolling = False) #showing Tableau dashboard hint from : https://towardsdatascience.com/embedding-tableau-in-streamlit-a9ce290b932b

  if __name__ == "__main__":    
    dashboard()

elif menu == 'Human Resource EDA':
  def app(title=None):
    st.title(title)

    data = pd.read_csv('HR.csv', index_col = False,  sep = ',', skipinitialspace = True)
    data['department'] = data['sales']
    data = data.drop(['sales'], axis=1)
    data['left'] = data['left'].astype(bool)
    st.write("First 5 rows of the dataset without any adjustment:")
    st.write(data.head())
    hrLeft = data['left'] == 1
    hrStay = data['left'] == 0
    Left = data[hrLeft]
    Stay = data[hrStay]

    st.write("## Human Resource Exploratory Data Analysis by SweetViz")
    
    comparison_report = sv.compare([Stay,'Stay'],[Left,'Left'])
    comparison_report.show_html(filepath='SWEETVIZ_REPORT.html', open_browser=True, layout='vertical', scale=1.0)
    HtmlFile = open("SWEETVIZ_REPORT.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code, width = 1150, height = 950, scrolling = True) #showing SweetViz hint from https://discuss.streamlit.io/t/include-an-existing-html-file-in-streamlit-app/5655/3
  

  app(title = 'Human Resource Exploratory Data Analysis')

  st.title("Machine Learning with the HR Dataset")

  data = pd.read_csv('HR.csv', index_col = False,  sep = ',', skipinitialspace = True)
  data['department'] = data['sales']
  data = data.drop(['sales'], axis=1)
  data['left'] = data['left'].astype(bool)
  data['salary'] = le.fit_transform(data['salary'].values)
  data['department'] = le.fit_transform(data['department'].values)

  X = data.drop('left', axis = 1)
  st.write(X.head())

  y = data['left']
  st.write(y.head())

  Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

  knn = KNeighborsClassifier()
  knn.fit(Xtrain, ytrain)
  ypred = knn.predict(Xtest)

  st.write( "## K-Nearest Neighbour")

  print(confusion_matrix(ytest, ypred))
  st.write("### K-NN Confusion Matrix: \n", (confusion_matrix(ytest, ypred)))
  print(classification_report(ytest, ypred))
  st.text((classification_report(ytest, ypred))) ##st.text hint from https://github.com/streamlit/streamlit/issues/1641

  RandomForest = RandomForestClassifier()
  RandomForest.fit(Xtrain, ytrain)
  ypred = RandomForest.predict(Xtest)

  st.write( "## Random Forest")

  print(confusion_matrix(ytest, ypred))
  st.write("### Random Forest Confusion Matrix: \n", (confusion_matrix(ytest, ypred)))
  print(classification_report(ytest, ypred))
  st.text((classification_report(ytest, ypred)))

  nb = GaussianNB()
  nb.fit(Xtrain, ytrain)
  ypred = nb.predict(Xtest)

  st.write( "## Naive Bayes")

  print(confusion_matrix(ytest, ypred))
  st.write("### Naive Bayes Confusion Matrix: \n", (confusion_matrix(ytest, ypred)))
  print(classification_report(ytest, ypred))
  st.text((classification_report(ytest, ypred)))

  st.title("Summary")
  st.write("""
  From the EDA result in SweetViz, for the employees that have **left** the company,
    - majority have a low satisfaction level which is in the range of 0.2 ~ 0.4.
    - majority have a low score for their last evaluation.
    - received a high number of projects done within their time in the company.
    - mostly have an average working hours of 100 ~ 150 or 250 ~ 300.
    - were employees that majority are experienced staffs of the company; working for more than 3 years in the company, 
      probably the majority that left found a better role in another company.
    - HR have the most employees that left, compared with the total number of employees by each department. Followed by Technical and Sales.
    - majority received a low salary
    - work accidents are not a factor of why employees left (low number of employees that have an accident at work)
  Something to note overall, **only ~3% of employees (stayed or left)** received a promotion within the last 5 years.

  From Machine Learning score result, **Random Forest** gave a very good score for this dataset.
  
  From the dataset features, it is apparent that the dataset is not for analyzing Employee Turnover rate or Attrition rate.
  It is more on **analysing the satisfaction level of employees working with the company**, ie. working hours, salary, career promotions.
  """)

elif menu == 'Human Resource Prediction':
  st.write("""
    # Human Resource Prediction
    This app shows the relationship between the features stated and to predict if the staff will leave or stay.
    Machine Learning model used Random Forest Classifier.
    Training data used is from [here](https://www.kaggle.com/colara/human-resource)
    """)
  def user_input_features():
      satisfaction_rate = st.slider('Satisfaction Rate', min_value = 0.0, max_value = 1.0)
      last_eval = st.slider('Last Evaluation', min_value = 0.0, max_value = 1.0)
      projectNo = st.text_input('Number of Project', value = 0, max_chars = 2, key = int)
      avgMonthlyHrs = st.text_input('Average Monthly Hours', value = 0, max_chars = 3, key = int)
      year_spend_company = st.text_input('Year with Company', value=0, max_chars = 2, key = int)
      promotion = st.checkbox("Have any promotion the last 5 years?")
      accident = st.checkbox("Have any accidents at work?")
      dept = st.selectbox("Department", ('Sales','Accounting',
                                       'Human Resource','Technical',
                                       'Support','Management','IT',
                                       'Product Management','Marketing','Others'))
      if dept == 'Sales': dept = 7
      elif dept == 'Accounting': dept = 2
      elif dept == 'Human Resource': dept = 3
      elif dept == 'Technical': dept = 9
      elif dept == 'Support': dept = 8
      elif dept == 'Management': dept = 4
      elif dept == 'Product Management': dept = 6
      elif dept == 'Marketing': dept = 5
      elif dept == 'Others' : dept = 1
    
      Salary = st.selectbox("Salary",('<2500','2600 - 10000','11000>'))
      if Salary == '<2500': Salary = 1
      elif Salary == '2600 - 10000': Salary = 2
      elif Salary == '11000>': Salary = 0

      answer = {
            'satisfaction_level': satisfaction_rate,
            'last_evaluation': last_eval,
            'number_project': projectNo,
            'average_montly_hours' : avgMonthlyHrs,
            'time_spend_company': year_spend_company,
            'promotion_last_5years': promotion,
            'work_accident' : accident,
            'salary': Salary,
            'department': dept}
      features = pd.DataFrame(answer, index=[0])
      return features

  df = user_input_features()
  df['salary'] = le.fit_transform(df['salary'].values)
  df['department'] = le.fit_transform(df['department'].values)

  st.subheader('User Input parameters')
  st.write(df)

  data = pd.read_csv('HR.csv', index_col = False,  sep = ',', skipinitialspace = True)
  data['department'] = data['sales']
  data = data.drop(['sales'], axis=1)
  data['left'] = data['left'].astype(bool)
  data['salary'] = le.fit_transform(data['salary'].values)
  data['department'] = le.fit_transform(data['department'].values)

  X = data.drop(['left'], axis = 1)
  Y = data['left']

  clf = RandomForestClassifier(max_depth=5, random_state=123)
  clf.fit(X, Y)

  prediction = clf.predict(df)
  prediction_proba = clf.predict_proba(df)

  st.subheader('Class labels and their corresponding index number')
  st.write('False(0) = Stay || True(1) = Leave')

  st.subheader('Prediction:')
  st.write(prediction)

  st.subheader('Prediction Probability:')
  st.write(prediction_proba)
