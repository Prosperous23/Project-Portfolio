import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#having a feel of our dataFrame/describing our data

housing = pd.read_csv(r"C:\Users\prosper\Downloads\Nashville-Housing.csv")
print(housing)
housingInfo = housing.info()
print(housingInfo)
housingShape = housing.shape
print(housingShape)
housingDtypes= housing.dtypes
print(housingDtypes)
lasttwenty = housing.tail(20)# gettting the last 20rows from our dataset
print(lasttwenty)

# Data Wrangling
housing = housing.rename(columns = {'UniqueID ':'UniqueID' }) # here we renamed a column in our dataFrame
housing = housing.drop(columns=['Unnamed: 0'])# dropping irrelevant columnn
print(housing.columns)
duplicates = housing.duplicated().sum()
print(f' we have {duplicates} duplicates')
emptyRows = housing.isnull().sum()#checking if we have null values
print(f' we have {emptyRows} empty/incomplete rows')

pd.set_option('display.max_columns',30)
#housing.loc[housing['SalePrice']=='$120,000', 'SalePrice']= '120000'
#housing.replace()
#housing['SalePrice'] = housing['SalePrice'].str.replace('$120000','120000')
housing['SalePrice'] = housing['SalePrice'].str.replace(',','')
housing['SalePrice'] = housing['SalePrice'].str.replace('$','')

u=housing['SalePrice'].iloc[162]
print(u)
#housing['SalePrice']=pd.to_numeric(housing['SalePrice'])
housing['SalePrice'] = housing['SalePrice'].astype('int64')
housing['SaleDate']= pd.to_datetime(housing['SalePrice'])
print(housing.dtypes)
good=housing[['ParcelID','SalePrice']]

# Data Mining and analysis
ShortDescriptiveAnalysis = housing.describe()
print(ShortDescriptiveAnalysis)
print(good.sort_values('SalePrice',ascending=False).agg(['mean','median','max','min']))
o=housing.groupby('Bedrooms')['SalePrice'].sum()

o.plot(kind= 'bar')
#plt.show()

sns.barplot(x=o.index,y=o.values)
#plt.show()
sns.catplot(data=housing,x='Bedrooms',kind='count')
#plt.show()
print(housing['Bedrooms'].value_counts())

print(housing)
housing['YearBuilt'] = housing['YearBuilt'].astype('str')

housing['YearBuilt']= housing['YearBuilt'].str.rstrip('0')
housing['YearBuilt']= housing['YearBuilt'].str.rstrip('.')
print(housing['YearBuilt'])
housing['YearBuilt']=pd.to_datetime(housing['YearBuilt'])
print(housing.dtypes)
# homes that were built after 1980 and expensive
def good_homes(X,Y):
    after = housing[(housing['YearBuilt']> X) & (housing['LandValue']>Y)]
    print(after)
good_homes('1980',30000)

housing[['Legal','Reference']] = housing['LegalReference'].str.split('-',1,expand=True)
housing['Legal_Reference'] = housing['Legal']+"-"+ housing['Reference']
housin = housing.info()
print(housin)

print(housing['YearBuilt'])

# Year with the highest number of houses built
housing['YearBuilt']= housing['YearBuilt'].dt.year
Date = housing.value_counts('YearBuilt')
print(Date)
# visual Representation
g=sns.catplot(data=housing, x= 'YearBuilt', kind ='count')
plt.xticks(rotation = 90)
plt.title('Year with the highest number of houses built')
#plt.show()
#HomeOwners of building with the highest Totalvalue
OwnerMax=housing.groupby('OwnerName')['TotalValue'].max()
print(OwnerMax)

# most Valued buildings
MostValuedBuildings = housing.sort_values('TotalValue', ascending= False).head(10)
print(MostValuedBuildings)

#Type of structure of houses and thier cost over the years
MoneySpent=housing.groupby('LandUse')['SalePrice'].sum().sort_values()
#from our analysis we can see people spent mostly on single Family homes
print(MoneySpent)
#sns.barplot(data=MoneySpent, x=MoneySpent.index, y=MoneySpent.values)
MoneySpent.plot(kind='bar')
#plt.show()
# Most type of houses built
SumSpent = housing.value_counts('LandUse')
print(SumSpent)
SumSpent.plot(kind='bar')
plt.title('Most type of houses built')
#plt.show()
# Companies/Organization who own more than 10 homes/Properties
Most_Owners = housing.value_counts('OwnerName').head(10)
print(Most_Owners)
Most_Owners.plot(kind ='bar')
#plt.title('People who own more than 10 homes')
#plt.show()


# ---------------- -----------Predictive Analysis Using Linear Regression(predicting the landValue from the SalePrice)----------------------------------
print(housing.dtypes)
# First we check corretion
correlation = housing.corr()
sns.heatmap(data=correlation,vmin=-1,vmax=1, annot= True)
plt.title('Correlation')
print(correlation)
#plt.show()
#---------------Relationship between SalePrice and TotalValue----------------
sns.relplot(data=housing, x= 'SalePrice', y=  'TotalValue',hue='TaxDistrict')
plt.title('Relationship between SalePrice and TotalValue')
#  plt.show()

#------------------Relationship between SalePrice and LandValue--------------------
sns.relplot(data=housing, x= 'SalePrice', y='LandValue')
#plt.show()
home =housing[['BuildingValue','LandValue','FullBath']]# selecting a dataframe for our model

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

train = home
test = housing.SalePrice
x_train, x_test, y_train,y_test = train_test_split(train, test, test_size = 0.3,random_state=2)

regr = LinearRegression()
rf=regr.fit(x_train,y_train)
print(rf)

pred = regr.predict(x_test)
print(pred)

RegrScore=regr.score(x_test,y_test)
print(RegrScore)






