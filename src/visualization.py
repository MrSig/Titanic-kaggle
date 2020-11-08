import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Visu:
    
    @staticmethod
    def heat_map(data, list_cols_you_dont_want, specific_feat = 'False'):
        
        """ Gives a heat map of attributes contained within the data passed as arg. Can specify if you dont want features to be considered """
        
        if specific_feat == 'True':
            dat_to_heat = data.drop(columns = list_cols_you_dont_want)
            
        if specific_feat == 'False':
            dat_to_heat = data
            
        fig, heatmap_plot = plt.subplots(figsize=(20,14))
        heatmap_plot = sns.heatmap(dat_to_heat.corr(), annot=True)
        return heatmap_plot
    
    @staticmethod
    def feat_hist(data):
        
        """ Gives histogram of attributes contained within the data passed as arg. """
        
        data_df = data.select_dtypes(exclude=['object'])
        nrows = len(data_df.columns.values)
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(6,20))

        for att in data_df.columns.values:
            data_df.loc[:,att].plot.hist(ax=axes[np.where(data_df.columns.values==att)[0][0]], legend = True)
        return print('done')
    
    @staticmethod
    def my_autopct(pct):
        return ('%.2f' % pct) if pct > 5 else ''
    
    @staticmethod
    def pie_chart(data, info=False):
        
        data = data.select_dtypes(exclude=['object'])
        for feat in data.columns:
            if feat != 'Survived':
                surv_cnt = data.groupby(feat).Survived.sum()
                cat_cnt = data.groupby(feat).apply(lambda df: df.loc[:,feat].count())
                surv_prop = surv_cnt.div(cat_cnt)
                labels = cat_cnt.index

                fig, ax = plt.subplots()
                plt.ylabel('Survived')
                plt.xlabel(feat)

                ax.pie(surv_prop, labels=labels)

                if info == True:
                    print(' \n ######### survivor count of',feat,'########## \n ')
                    print(surv_cnt)
                    print(' \n ######### category count of',feat,'########## \n ')
                    print(cat_cnt)
                    print(' \n ######### proportion of survivors',feat,'########## \n ')
                    print(surv_prop) 
                else:
                    continue
                    