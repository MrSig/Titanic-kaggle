import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


class Impute:
    
    @staticmethod
    def x_to_pred(data, tuple_explanatory, str_y):
        
        """ Outputs the explanatory var. to be used to predict y."""

        x_p = data.loc[data.loc[(data.loc[:,str_y].isnull())].index.values]
        x_to_pred = x_p.loc[:,tuple_explanatory].values

        return x_to_pred
    
    
    @staticmethod
    def reg(data, n_splits, tuple_explanatory, str_y):

        """ Outputs a trained regression of str_y on the explanatory att. """
        
        train_data = data.drop(data.loc[(data.loc[:,str_y].isnull())].index.values)

        x_dat = train_data.loc[:,tuple_explanatory].values
        y_dat = train_data.loc[:,str_y].values
        
        kf = KFold(n_splits=n_splits)
        in_rscore = 0
        out_rscore = 0
        for train_index, test_index in kf.split(y_dat):

            x_train, y_train = x_dat[train_index], y_dat[train_index]
            x_test, y_test = x_dat[test_index], y_dat[test_index]

            model = LinearRegression().fit(x_train, y_train)

            in_rscore_n = model.score(x_train, y_train)
            in_rscore += in_rscore_n/n_splits

            out_rscore_n = model.score(x_test, y_test)
            out_rscore += out_rscore_n/n_splits

        return model, print( ' \n\n##### Lin. Reg. on ', str_y ,' #####\n ' ,' \n\n##### COEFFICIENTS #####\n ', 
                            model.coef_,' \n\n##### IN RSCORE #####\n ' ,in_rscore,' \n\n##### OUT RSCORE #####\n ', out_rscore)
    
    
    @staticmethod
    def knn(data, n_neighbors, tuple_explanatory, str_y):

        """ Outputs a knn classifier of str_y based on chosen explanatory att. """
        
        train_data = data.drop(data.loc[(data.loc[:,str_y].isnull())].index.values)

        x_dat = train_data.loc[:,tuple_explanatory].values
        y_dat = train_data.loc[:,str_y].values
        
        model = KNeighborsClassifier(n_neighbors=n_neighbors).fit(x_dat, y_dat)
        score = model.score(x_dat, y_dat)
        
        return model, print( ' \n\n##### knn on ', str_y ,' #####\n ' ,' \n\n##### attributes used #####\n ', 
                            tuple_explanatory,' \n\n##### SCORE #####\n ', score )
    
    @staticmethod
    def replace(data, y_pred, str_y):
        
        """ Replaces missing values in given data with a predicted y """

        null_idx = data.loc[pd.isnull(data.loc[:,str_y])].index
        pred_series = pd.Series((y_pred), index = null_idx)
        data_fill = data.loc[:,str_y].fillna(pred_series)
        data.loc[:,str_y] = data_fill

        return data