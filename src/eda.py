import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging


class EDA:
    # intialize the dataframe
    def __init__(self, path):
        self.info_log = logging.getLogger('info')
        self.info_log.setLevel(logging.INFO)

        info_handler = logging.FileHandler('..\logs\info_log')
        info_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        info_handler.setFormatter(info_formatter)
        self.info_log.addHandler(info_handler)

        self.error_log = logging.getLogger('error')
        self.error_log.setLevel(logging.ERROR)

        error_handler = logging.FileHandler('..\logs\errors.log')
        error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        error_handler.setFormatter(error_formatter)
        self.error_log.addHandler(error_handler)
        # self.df = pd.read_csv(path)
        # print('b')
        try :
            self.df = pd.read_csv(path)
            self.info_log('Loading the file')
        except:
            self.error_log.error("Error occurred when loading the files")

    # return the dataframe
    def get_dataframe(self):
        self.info_log.info('Return the dataframe')
        return self.df
    
    # unique number in each columns
    def over_view(self):
        self.info_log.info('Over view information on the dataset')
        # shape of the dataframe
        print(f"Shape: {self.df.shape}")
        # information
        print('Information on each columns')
        print(self.df.info())

        # general statistic of the numerical columns
        print('Statistics:')
        print(self.df.describe())

    # number of unique values in each columns
    def unique_values(self):
        self.info_log.info('Unique values of each features')
        columns = self.df.columns.tolist()

        for column in columns:
            print(f'{column} : {self.df[column].nunique()}')

    # change datatype
    def change_datatype(self):
        self.info_log.info('Change datatype of some columns to category')

        # remove country and currency code
        self.df = self.df.drop(columns=['CurrencyCode','CountryCode'], axis = 1)

        # conver FraudResult and PricingStrategy to category
        self.df['FraudResult'] = self.df['FraudResult'].astype('category')
        self.df['PricingStrategy'] = self.df['PricingStrategy'].astype('category')

        object_columns = self.df.select_dtypes('object').columns.tolist()

        for obj_col in object_columns:
            if obj_col not in ['TransactionId','BatchId','TransactionStartTime']:
                self.df[obj_col] = self.df[obj_col].astype('category')

            elif obj_col == 'TransactionStartTime':
                self.df[obj_col] = pd.to_datetime(self.df[obj_col])

            else: pass

        print(f"New shape of the dataframe: {self.df.shape}")
        print(f"Information on the new dataset: {self.df.info()}")

    # distribution of features
    def distribution_stats_category(self):
        self.info_log.info('Catagorical datatype distribution plot')
        category_columns = ['ProviderId','ProductId','ProductCategory','ChannelId','PricingStrategy','FraudResult']

        fig, axs = plt.subplots(3,2, figsize=(19,16))
        axs = axs.flatten()

        for i,col in enumerate(category_columns):
            sns.countplot(data=self.df, x = col, ax=axs[i])
            axs[i].set_title('Distribution of ' + col)
            axs[i].set_ylabel('Count')
            
            if col == 'ProductId':
                axs[i].tick_params(axis='x', rotation=45) 
            else:
                axs[i].tick_params(axis='x', rotation=15) 

        plt.tight_layout()
        plt.show()

    def distribution_stats_numerical(self):
        self.info_log.info('Numrical datatype distribution plot')
        number_columns = ['Amount','Value']

        fig, axs = plt.subplots(1,2, figsize=(10,5))
        axs = axs.flatten()

        for i,col in enumerate(number_columns):
            sns.boxplot(data = self.df, x=col, ax = axs[i])
            # axs[i].hist(x = col, bins=2)
            axs[i].set_title('Distribution of ' + col)
            axs[i].set_ylabel('count')


        plt.tight_layout()
        plt.show()

    # correlation between numerical features
    def corr_relation(self):
        self.info_log.info('Correlation Analaysis of numerical features')
        corr = self.df[self.df.select_dtypes(include=['int64', 'float64']).columns].corr()
        print(corr)

    # analysis of features
    def feature_analysis(self, num):
        if num == 1:
            print(self.df.groupby('BatchId')['CustomerId'].nunique().sort_values(ascending=False))

        elif num == 2:
            print(f"Maximum number of transaction per batch: {self.df.groupby('BatchId')['TransactionId'].count().max()}")  
            print(f"Minimum number of transaction per batch: {self.df.groupby('BatchId')['TransactionId'].count().min()}")   

        elif num == 3:
            print(f"Maximum number of transaction per customer: {self.df.groupby('CustomerId')['TransactionId'].count().max()}")  
            print(f"Minimum number of transaction per customer: {self.df.groupby('CustomerId')['TransactionId'].count().min()}")   
        
        elif num == 4:
            print(f"Maximum number of customers per account: {self.df.groupby('AccountId')['CustomerId'].nunique().max()}")  
            print(f"Minimum number of customers per account: {self.df.groupby('AccountId')['CustomerId'].nunique().min()}") 

        elif num == 5:
           print(self.df.groupby('AccountId')[['BatchId','CustomerId','SubscriptionId']].nunique().sort_values(by = 'BatchId',ascending=False))      
        
        elif num == 6:
           columns = ['ProviderId','ProductId','ProductCategory', 'ChannelId']

           for col in columns:
               new_df = self.df.groupby(col)['CustomerId'].nunique().reset_index()
               plt.bar(new_df[col], new_df['CustomerId'])
               # Plot a bar chart
               plt.xlabel(col)
               plt.ylabel('Unique Customer Count')
               plt.title(f'Unique Customers by {col}')
               plt.xticks(rotation=45)  # Rotate x-ticks if necessary for better readability
               plt.tight_layout()  # Adjust layout to prevent overlap
               plt.show()

        elif num == 7:
            print(self.df.groupby('ProviderId')[['ProductId','ProductCategory']].nunique())       

        else: pass   



    # close log process
    def close_log(self):
        handlers = self.info_log.handlers[:]

        # close info logging 
        for handler in handlers:
            handler.close()
            self.info_log.removeHandler(handler)

        # close error logging
        handlers = self.error_log.handlers[:]
        for handler in handlers:
            handler.close()
            self.error_log.removeHandler(handler)
        
    
