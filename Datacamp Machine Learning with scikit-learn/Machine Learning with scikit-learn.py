## 1. Classification
### Which of these is a classification problem?
"""Once you decide to leverage supervised machine learning to solve a new problem, 
you need to identify whether your problem is better suited to classification or regression. 
This exercise will help you develop your intuition for distinguishing between the two.
"""
#Using labeled financial data to predict whether the value of a stock will go up or go down next week.

### Exploratory data analysis
import pandas as pd 

df=  pd.read_csv('house-votes-84.data', sep="," , header = None) # reading csv files
df.rename(columns = {0: 'Party',1:'infants', 2:'water', 3:'budget', 4: 'physician', 5:'salvador', #create the column names
      6:'religious', 7:'satellite', 8:'aid', 9:'missile', 10:'immigration', 11:'synfuels',
       12:'education', 13:'superfund', 14:'crime', 15:'duty_free_exports', 16:'eaa_rsa' 
    }, inplace = True)

df = df.replace({'y': 1, 'n': 0, '?':0}) # need to convert y = 1, n = 0, ? = 0                                                                                        

print(df.head()) #first 5 obs
print(df.info()) #info about columns, non-null count, dtype
print(df.describe()) #gives count, unique val, top val, freq
print(df.shape) #gives rows vs columns

#possible answers

#df has total of 435 rows and 17 columns - True
#except for party, all columns are type int 64 - 
#first 2 rows are votes made by republicans and next 3 are for democrats - True
#there are 17 predictor variables - False - there is 16 predictor and 1 target variable
#target variable is party - true


## 2. Regression




## 3. Fine Tuning your model















## 4. Preprocessing and Pipeline

 
