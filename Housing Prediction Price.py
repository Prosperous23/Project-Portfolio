#!/usr/bin/env python
# coding: utf-8

# IMPORT LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats
import pylab


# IMPORT DATA

# In[2]:


df = pd.read_csv(r"C:\Users\User\Downloads\kc-house-data (1).csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.isna().sum()


# In[7]:


df[df.duplicated]


# In[8]:


distinct_counts = []
for column in df.columns:
    distinct_count = df[column].nunique()
    first_5_unique_values = df[column].unique()[:5]
    last_5_unique_values = df[column].unique()[-5:]
    
    distinct_counts.append({'Column': column, 
                           'Distinct_Values_count': distinct_count,
                            'First_5_unique_values' : first_5_unique_values,
                            'Last_5_unique_values': last_5_unique_values})
    
distinct_count_df = pd.DataFrame(distinct_counts)
distinct_count_df.sort_values(by='Distinct_Values_count',ascending = False, ignore_index = True)
    


# In[9]:


df.describe()


# In[10]:


df.date = pd.to_datetime(df['date'])
df.dtypes


# In[11]:


numeric_columns = ['sqft_lot', 'sqft_lot15', 'sqft_living','sqft_living15','sqft_above',
                   'sqft_basement','lat','long','yr_built','yr_renovated']


# In[12]:


correlation = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(correlation,annot = True, cmap = 'Spectral')
plt.show()


# In[13]:


plt.figure(figsize=(20,10))
sns.histplot(df, x= 'price',kde = True)
plt.xlabel('PRICE ($)')
plt.ylabel('FREQUENCY')
plt.title('Price Distribution')
plt.show()


# In[14]:


def num_combined_plot(data, x, y):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot the histogram with KDE
    sns.histplot(data=data, x=x, kde=True, ax=axes[0], color='coral')
    
    # Plot the scatterplot with a correlation line
    sns.regplot(data=data, x=x, y=y, ax=axes[1], color='teal', 
                scatter_kws={'edgecolor': 'white'}, line_kws={"color": "coral"})

    # Calculate the correlation coefficient
    corr_coeff = data[[x, y]].corr().iloc[0, 1]

    # Annotate the correlation coefficient on the scatter plot
    axes[1].annotate(f'Correlation : {corr_coeff:.2f}', xy=(0.65, 0.9), xycoords='axes fraction', fontsize=14, color='coral')

    # Adjust plot aesthetics
    sns.despine(bottom=True, left=True)
    axes[0].set(xlabel=f'{x}', ylabel='Frequency', title=f'{x} Distribution')
    axes[1].set(xlabel=f'{x}', ylabel=f'{y}', title=f'{x} vs {y}')
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()

    plt.show()


# In[15]:


num_combined_plot(df, 'sqft_lot', 'price')


# In[16]:


num_combined_plot(df, 'sqft_lot15', 'price')


# In[17]:


num_combined_plot(df, 'sqft_living15', 'price')


# In[18]:


num_combined_plot(df, 'sqft_living', 'price')


# In[19]:


num_combined_plot(df, 'sqft_above', 'price')


# In[20]:


num_combined_plot(df, 'sqft_basement', 'price')


# In[21]:


def create_subplot_grid(data, x, y):
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the countplot with percentages
    sns.countplot(data=data, x=x, ax=axes[0], palette='Set2')
    axes[0].set(title=f'{x} Frequency')
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].set_ylabel('Count (%)')
    
    # Calculate and annotate the percentages
    total = len(data)
    for p in axes[0].patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x_ = p.get_x() + p.get_width() / 2
        y_ = p.get_height()
        axes[0].annotate(percentage, (x_, y_), ha='center', va='bottom')

    # Plot the boxplot
    sns.boxplot(data=data, x=x, y=y, ax=axes[1], palette='Set2')
    axes[1].set(title=f'Price vs. {x}')
    axes[1].tick_params(axis='x', rotation=90)

    # Plot the scatterplot with colors based on x
    sns.scatterplot(data=data, x=x, y=y, ax=axes[2], hue=x, palette='Set2')
    axes[2].set(title=f'{y} vs. {x}')
    axes[2].tick_params(axis='x', rotation=90)
    axes[2].yaxis.set_label_position("right")
    
    # Add the regression line to the scatterplot
    sns.regplot(data=data, x=x, y=y, ax=axes[2], color='coral', scatter=False)
    axes[2].get_legend().remove()
    
    plt.tight_layout()
    plt.show()


# In[22]:


create_subplot_grid(df, 'bedrooms','price')


# In[23]:


create_subplot_grid(df, 'floors','price')


# In[24]:


create_subplot_grid(df, 'bathrooms','price')


# In[25]:


create_subplot_grid(df, 'view','price')


# In[26]:


create_subplot_grid(df, 'waterfront','price')


# In[27]:


df['view'].value_counts()


# In[28]:


df['waterfront'].value_counts()


# In[29]:


df[df['waterfront']=='Yes']=1
df[df['waterfront']=='No']=0


# In[30]:


df['waterfront'].value_counts()


# In[31]:


create_subplot_grid(df, 'condition','price')


# In[32]:


create_subplot_grid(df, 'grade','price')


# In[33]:


# year built vs price TREND, how it relates with one another 
plt.figure(figsize=(15,5))
sns.lineplot(data = df, x = 'yr_built',y = 'price',marker= 'o',markerfacecolor = 'red')
plt.title('Price depending on year built')
plt.show()


# In[34]:


#what year was the most expensive house built and how much?
df.groupby('yr_built')['price'].max().sort_values(ascending= False)
df.groupby('yr_built')['price'].max().sort_values(ascending= False)


# In[35]:


#what year made the most money
df.groupby('yr_built')['price'].sum().sort_values(ascending= False)


# In[36]:


#what year sold most houses
df['yr_built'].value_counts()


# In[37]:


df['yr_renovated'].value_counts()
renovated_count =len(df[df['yr_renovated']!=0])
print(renovated_count)


# In[38]:


non_renovated = 20699/len(df) * 100


# In[39]:


non_renovated


# In[40]:


renovated = 914/len(df)* 100


# In[41]:


renovated


# In[42]:


sizes = [renovated,non_renovated]
labels = ['renovated', 'not renovated']


# In[43]:


sizes


# In[44]:


plt.pie(sizes, labels=labels,autopct ='%1.1f%%',startangle = 0)
plt.title('percentage of renovated against not renovated')
plt.show()


# In[45]:


sns.scatterplot(data =df, x='sqft_living',y='price',hue='grade')
plt.xlabel('living space (sqft)')
plt.ylabel('price')
plt.title('scatter plot of price vs living space by grade')
plt.legend(title='Grade')
plt.show()


# In[46]:


sns.scatterplot(data =df, x='sqft_living',y='price',hue='bathrooms')
plt.xlabel('living space (sqft)')
plt.ylabel('price')
plt.title('scatter plot of price vs living space by grade')
plt.legend(title='bathroom')
plt.show()


# In[47]:


df.iloc[:,1:19]


# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


X = df[['sqft_living','sqft_living15','sqft_above']]
y = df['price']


# In[50]:


y


# In[51]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[52]:


X_train.shape, X_test.shape,y_train.shape,y_test.shape


# In[53]:


from sklearn import preprocessing,svm
from sklearn.linear_model import LinearRegression


# In[54]:


lr = LinearRegression()


# In[55]:


lr.fit(X_train,y_train)


# In[56]:


lr.score(X_test,y_test)


# In[57]:


y_lr_train_pred = lr.predict(X_train)


# In[58]:


y_lr_test_pred = lr.predict(X_test)


# In[59]:


from sklearn.metrics import mean_squared_error,r2_score


# In[60]:


lr_train_mse = mean_squared_error (y_train,y_lr_train_pred)


# In[61]:


lr_test_mse = mean_squared_error(y_test,y_lr_test_pred)


# In[62]:


lr_test_mse


# In[63]:


df


# In[64]:


df['month']= df['date'].dt.month


# In[65]:


df.sort_values(by = 'month', ascending = True)


# In[66]:


plt.figure=(figsize(30,20))
sns.lineplot(data = df, x = 'date', y ='price')
plt.show()

