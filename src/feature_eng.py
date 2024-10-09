import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from optbinning import BinningProcess,OptimalBinning
from sklearn.model_selection import train_test_split


class FeatEng:
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
    
    # change datatype
    def change_datatype(self):
        self.info_log.info('Change datatype of some columns to category')

        # remove country and currency code
        self.df = self.df.drop(columns=['CurrencyCode','CountryCode'], axis = 1)

        # conver FraudResult and PricingStrategy to category
        # self.df['FraudResult'] = self.df['FraudResult'].astype('category')
        self.df['PricingStrategy'] = self.df['PricingStrategy'].astype('category')

        object_columns = self.df.select_dtypes('object').columns.tolist()

        for obj_col in object_columns:
            if obj_col == 'TransactionStartTime':
                self.df[obj_col] = pd.to_datetime(self.df[obj_col]).dt.tz_localize(None)

            else: 
                self.df[obj_col] = self.df[obj_col].astype('category')

        print(f"New shape of the dataframe: {self.df.shape}")
        print(f"Information on the new dataset: {self.df.info()}")

    # add total transaction column
    def aggregate_transaction(self):
        self.info_log.info('Feature engineering aggregating over customers')
        # monetary
        new = self.df.groupby('CustomerId')['Amount'].sum().reset_index()
        self.df = self.df.merge(new, on = 'CustomerId', how='left', suffixes=('','_Total'))

        new = self.df.groupby('CustomerId')['Amount'].mean().reset_index()
        self.df = self.df.merge(new, on = 'CustomerId', how='left', suffixes=('','_Average'))

        new = self.df.groupby('CustomerId')['Amount'].std().reset_index()
        self.df = self.df.merge(new, on = 'CustomerId', how='left', suffixes=('','_std'))

        # frequency
        new = self.df.groupby('CustomerId')['TransactionId'].count().reset_index()
        self.df = self.df.merge(new, on = 'CustomerId', how='left', suffixes=('','_frequency'))

        # recency
        recency_df = self.df.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
        today = pd.to_datetime('2024-01-01').tz_localize(None)

        # the number of days since the latest transaction
        recency_df['Recency'] = (today - recency_df['TransactionStartTime']).dt.days
        self.df = self.df.merge(recency_df[['CustomerId', 'Recency']], on = 'CustomerId', how='left')

        # severity
        severity_df = self.df.groupby('CustomerId')['FraudResult'].sum().reset_index()
        self.df = self.df.merge(severity_df, on = 'CustomerId', how='left',suffixes=('','_severity'))

        print(self.df.shape)

    # extract year, month, day
    def extract_datetime(self):
        self.info_log.info('Extracting date feature from datetime')
        self.df['Month'] = self.df['TransactionStartTime'].dt.month
        self.df['Year'] = self.df['TransactionStartTime'].dt.year
        self.df['Day'] = self.df['TransactionStartTime'].dt.day
        self.df['Hour'] = self.df['TransactionStartTime'].dt.hour
        
        print(f'Shape of the dataframe {self.df.shape}')


    # categorical features to numerical
    def label_encoder(self):
        self.info_log.info('Label encoding categorical features')
        encoder = LabelEncoder()
        categorical_col = self.df.select_dtypes(include='category').columns.tolist()

        for col in categorical_col:
            self.df[col] = encoder.fit_transform(self.df[col])

        print(f'Columns informatio: {self.df.info()}') 

    # checking null numbers
    def null_col(self):
        self.info_log.info('Filling null values by zero')
        for col in self.df.columns:
            if self.df[col].isnull().any() == True:
                print(f'Column containing null value: {col}')
                self.df[col].fillna(0, inplace=True)

    # RFMS
    def risk_score(self):
        self.info_log.info('Calculating risk score')
        self.df['Risk_Score'] = self.df['Recency'] + self.df['TransactionId_frequency'] + self.df['Amount_Average'] + self.df['FraudResult_severity']

    # normalization
    def normalization(self):
        self.info_log.info('Standardization using standard scaler')
        columns = ['Amount_Total','Amount_Average','Amount_std','TransactionId_frequency','Recency']
        scaler = StandardScaler()
        for col in columns:
            self.df[col] = scaler.fit_transform(self.df[[col]])

    # clusters 
    def risk_space(self):
        self.info_log.info('Using k-means for getting default behavior')
        # Perform KMeans clustering on RFMS space
        kmeans = KMeans(n_clusters=2, random_state=42)  # 2 clusters (good, bad)
        self.df['Cluster'] = kmeans.fit_predict(self.df[['Recency', 'TransactionId_frequency', 'Amount_Average', 'FraudResult_severity']])

        print(kmeans.cluster_centers_)

    # building woe
    def woe_best_festures(self):
        self.info_log.info('Weight of Evidence')
        # differentiate features to continuous and categorical for woe implmentation 
        continuous_feat = ['Amount_Total','Amount_Average','Amount_std','TransactionId_frequency','FraudResult_severity','Amount','Value','Risk_Score']
        categorical_feat = ['TransactionId','BatchId','AccountId','SubscriptionId','CustomerId','ProviderId','ProductId','ProductCategory','ChannelId','PricingStrategy','FraudResult','Month','Year','Day','Hour','Cluster']

        def woe_iv_cont(df, feature, max_bins=6):
            optb = OptimalBinning(name=feature ,dtype='numerical', max_n_bins=max_bins)
            optb.fit(df[feature],df['Cluster'])

            # transform feature to WoE values
            woe_values = optb.transform(df[feature], metric="woe")
            # calculate Iv
            binning_table = optb.binning_table.build()
            iv = binning_table.loc['Totals', 'IV']

            return woe_values, iv
        
        def woe_iv_cate(df, feature):
            optb = OptimalBinning(name=feature,dtype='categorical')
            optb.fit(df[feature],df['Cluster'])

            # transform feature to WoE values
            woe_values = optb.transform(df[feature], metric="woe")

            # calculate Iv
            binning_table = optb.binning_table.build()
            iv = binning_table.loc['Totals', 'IV']

            return woe_values, iv
        
        # feature selection 
        important_continous_features = []
        important_categorical_features = []

        # threshold value of iv
        threshold_cont = 0.05
        threshold_cate = 0.1

        for feat in continuous_feat:
            woe_values, iv = woe_iv_cont(self.df, feat)
            if iv >= threshold_cont:
                important_continous_features.append(feat)

            self.df[feat + '_woe'] = woe_values

        for feat in categorical_feat:
            woe_values, iv = woe_iv_cate(self.df, feat)
            if iv >= threshold_cate:
                important_categorical_features.append(feat)

            self.df[feat + '_woe'] = woe_values

        selected_features = important_continous_features + important_categorical_features

        print("Selected Continuous Features based on IV:", important_continous_features)
        print("Selected Categorical Features based on IV:", important_categorical_features)
        print("All Selected Features:", selected_features)  


    # close logging
    def close_log(self):
        self.info_log.info('Closing logging')
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








        

        
        






            



