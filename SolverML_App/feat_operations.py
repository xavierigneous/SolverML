import pandas as pd
import pandas.io.sql as psql
import numpy as np
import os, time
from sklearn.exceptions import NotFittedError
import scipy.stats as stats

from db_operations import sql_engine, get_current_project

temp_dir = os.path.join(os.getcwd(), 'temp_data')

def symbol_remove(x):
    x = x.replace('\ |\?|\,|\!|\/|\;|\:|\$', '', regex=True)
    x = x.astype(float)
    return x
class type_change:
    def to_float(x):
        x = x.replace("\ |\?|\,|\!|\/|\;|\:|\$|\'", '', regex=True)
        x = x.replace(r'', np.nan, regex=True)
        x = x.astype(float)
        return x
    def to_int(x):
        #x = x.replace('\ |\?|\,|\!|\/|\;|\:|\$', '', regex=True)
        x = x.replace(r'', np.nan, regex=True)
        x = x.astype(int)
        return x
    def to_string(x):
        #x = x.replace(r'', np.nan, regex=True)
        x = x.astype(str)
        return x

class transformations:
    def log_transform(x):
        return np.log1p(x)

    def log_inverse_transform(x):
        return np.exp(x) - 1
    
    def square_transform(x):
        return np.log1p(x)
    
    def box_cox_transform(x):
        return stats.boxcox(x)


def datetimeformat(feats):
    return pd.to_datetime(data_in.loc[:, feats])


def onehotencode(data_in, feats):
    return pd.get_dummies(data_in, columns=feats)



def reverse_one_hot(df, cols):
    for col in cols:
        print('Removing : ', col)
        unencode = pd.DataFrame(
            [x for x in np.where(df.filter(like=col) == 1, df.filter(like=col).columns, '').flatten().tolist() if
             len(x) > 0], columns=[col])
        df = df.drop(df.filter(like=col), axis=1)
        df = pd.concat([df, unencode], axis=1, ignore_index=False)
    return df


def store_temp_operation(operation, cols):
    feat_engine = pd.DataFrame(index=range(0), columns=['Operation', 'Column', 'Key'])
    print(type(operation).__name__)
    if type(operation).__name__=='ColumnTransformer':
        print('TEMP ',type(operation).__name__)
        key=[operation.transformers[0][0] + '_' + x for x in cols if not str(x) == "nan"]
        print(key)
        feat_engine = feat_engine.append({'Operation': operation, 'Column': cols, 'Key':[key]}, ignore_index=True)
    elif type(operation).__name__=='function':
        print(operation.__name__)
        key=[operation.__name__ + '_' + x for x in cols if not str(x) == "nan"]
        print(key)
        feat_engine = feat_engine.append({'Operation': operation, 'Column': cols, 'Key':key}, ignore_index=True)
    elif type(operation).__name__=='defaultdict':
        print(operation.default_factory.__name__)
        key=[operation.default_factory.__name__ + '_' + x for x in cols if not str(x) == "nan"]
        print(key)
        feat_engine = feat_engine.append({'Operation': operation, 'Column': cols, 'Key':key}, ignore_index=True)
    else:
        feat_engine = feat_engine.append({'Operation': operation, 'Column': cols}, ignore_index=True)

    
    print(feat_engine)
    feat_engine.to_pickle(os.path.join(temp_dir, 'temp_operations.pkl'))


def apply_operations(test_data, feat_engine):
    print(feat_engine)
    for operation, column in zip(feat_engine['Operation'], feat_engine['Column']):
        print(operation, ' on ', column)
        print(test_data.loc[:, column].dtypes)
        print(type(operation).__name__)
        if type(operation).__name__ == 'ColumnTransformer':
            print('Operation: ', operation.transformers[0][1][0])
            try:
                test_data.loc[:, column] = test_data.loc[:, column].apply(operation.fit_transform)
            except:
                test_data.loc[:, column] = operation.fit_transform(test_data.loc[:, column])
            print(test_data.loc[:, column])
        elif type(operation).__name__ == 'function':
            print(operation.__name__)
            if operation.__name__ == 'onehotencode':
                test_data = onehotencode(test_data, column)
                print(test_data)
            else:
                print(operation)
                print('result ', test_data.loc[:, column].apply(operation))
                test_data.loc[:, column] = test_data.loc[:, column].apply(operation)
                # print(operation)
                print(test_data.loc[:, column].dtypes)
        elif type(operation).__name__ == 'defaultdict':
            type(operation)
            print('result ')
            print(test_data.loc[:, column].apply(lambda x: operation[x.name].fit_transform(x)))
            test_data.loc[:, column] = test_data.loc[:, column].apply(lambda x: operation[x.name].fit_transform(x))
            print(test_data.loc[:, column].dtypes)
    return test_data


def apply_reverse_operations(user_name, test_data, feat_engine):
    print(feat_engine)
    for operation, column in zip(feat_engine['Operation'], feat_engine['Column']):
        print(operation, ' on ', column)
        print(type(operation).__name__)
        if type(operation).__name__ == 'ColumnTransformer':
            print('Operation: ', operation.transformers[0][1][0])
            if operation.transformers[0][0] in ['_int','_float','_string','remove_outliers','median_impute_outliers','mean_impute_outliers', '_log', '_sqrt',\
                                                '_boxcox']:
                print('Applying data reversal: ',operation.transformers[0][0])
                project_id=get_current_project(user_name)
                query=f"select temp from public.train_data where user_name='{user_name}' AND project_id='{project_id}' AND use_file='Yes'".format(user_name, project_id)
                # df=pd.read_json(sql_engine()(query, return_data=True)['temp'][0])
                df=pd.read_json(psql.read_sql_query(query, sql_engine())['temp'][0])
                print('temp data',df)
                #df=pd.read_json(data['temp'][0])
                test_data.loc[:,column]=df[column]
            else:    
                try:
                    test_data.loc[:, column] = test_data.loc[:, column].apply(
                        operation.transformers[0][1][0].inverse_transform)
                except NotFittedError:
                    print('This Operation has no inverse function')
                    project_id=get_current_project(user_name)
                    query=f"select temp from public.train_data where user_name='{user_name}' AND project_id='{project_id}' AND use_file='Yes'".format(user_name, project_id)
                    start_time = time.time()
                    data = psql.read_sql_query(query, sql_engine())
                    # data=sql_engine()(query, return_data=True)
                    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
                    df=pd.read_json(data['temp'][0])
                    test_data.loc[:,column]=df[column]

                except Exception as ex:
                    print('temp error ',ex)
                    test_data.loc[:, column] = operation.transformers[0][1][0].inverse_transform(test_data.loc[:, column])
                print(test_data.loc[:, column])
        elif type(operation).__name__ == 'function':
            print(operation.__name__)
            if operation.__name__ in ['onehotencode','drop_columns']:
                print(operation.__name__,' reverse')
                if operation.__name__ == 'onehotencode':
                    test_data = reverse_one_hot(test_data, column)
                project_id=get_current_project(user_name)
                query=f"select temp from public.train_data where user_name='{user_name}' AND project_id='{project_id}' AND use_file='Yes'".format(user_name, project_id)
                # df=pd.read_json(sql_engine()(query, return_data=True)['temp'][0])
                df=pd.read_json(psql.read_sql_query(query, sql_engine())['temp'][0])
                #df=pd.read_json(data['temp'][0])
                test_data.loc[:,column]=df[column]
            else:
                print(operation)
                print('result ', test_data.loc[:, column].apply(operation))
                test_data.loc[:, column] = test_data.loc[:, column].apply(operation)
                # print(operation)
                print(test_data.loc[:, column].dtypes)
        elif type(operation).__name__ == 'defaultdict':
            type(operation)
            print('result ')
            print(test_data.loc[:, column].apply(lambda x: operation[x.name].inverse_transform(x)))
            test_data.loc[:, column] = test_data.loc[:, column].apply(lambda x: operation[x.name].inverse_transform(x))
            print(test_data.loc[:, column].dtypes)
    return test_data

def drop_columns(data_in,cols):
    return data_in.drop(cols,axis=1)
def remove_outlier_IQR(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    df_final=df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)))]
    return df_final