import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer



class Utils:
    
    @staticmethod
    def binning(data, tuple_want_bin):
        
        """ Bins the features given as arg. into ten bins. """
        
        bin_est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
        dat_to_bin = data.loc[:,tuple_want_bin].dropna()

        binned_arr = bin_est.fit_transform(dat_to_bin)
        binned_dat = pd.DataFrame(binned_arr, index = data.loc[:,tuple_want_bin].dropna().index, columns =tuple_want_bin ) 
        data.loc[:,tuple_want_bin] = binned_dat.loc[:,tuple_want_bin]

        return data
    
    @staticmethod     
    def map_apply(row):
        
        """ Custom function to be used in conjuction with pandas's apply() method.  """

        if ('Mr.' in row.Name)==True:
            row.Name = 'mr'
            
        if ('Mrs.' in row.Name)==True or ('Mme.' in row.Name)==True or ('Lady' in row.Name)==True or ('Countess' in row.Name)==True:
            row.Name = 'mrs'
            
        if ('Miss.' in row.Name)==True or ('Ms.' in row.Name)==True or ('Mlle.' in row.Name)==True:
            row.Name = 'miss'
            
        if ('mr' in row.Name)==False and ('mrs' in row.Name)==False and ('miss' in row.Name)==False:
            row.Name = 'staff'
        
        if row.Cabin == str(row.Cabin) and ('A' in row.Cabin)==True:
            row.Cabin = 'A'

        if row.Cabin == str(row.Cabin) and ('B' in row.Cabin)==True:
            row.Cabin = 'B'

        if row.Cabin == str(row.Cabin) and ('C' in row.Cabin)==True:
            row.Cabin = 'C'

        if row.Cabin == str(row.Cabin) and ('D' in row.Cabin)==True:
            row.Cabin = 'D'

        if row.Cabin == str(row.Cabin) and ('E' in row.Cabin)==True:
            row.Cabin = 'E'

        if row.Cabin == str(row.Cabin) and ('F' in row.Cabin)==True:
            row.Cabin = 'F'

        if row.Cabin == str(row.Cabin) and ('G' in row.Cabin)==True:
            row.Cabin = 'G'

        if row.Cabin == str(row.Cabin) and ('T' in row.Cabin)==True:
            row.Cabin = 'T'

        return row