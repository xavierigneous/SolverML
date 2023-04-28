import pandas as pd
import pandas.io.sql as psql
import numpy as np
import sqlalchemy
import os, pickle, compress_pickle, sys
import time

temp_dir = os.path.join(os.getcwd(), 'SolverML_App/temp_data')

def sql_engine():
    sql_engine = sqlalchemy.create_engine('postgresql://postgres:vish2303@localhost:5432/analytics1')
    return sql_engine


#get all projects of a username
def get_projects(user_name):
    start_time = time.time()
    query = "select * from base_data where user_name='" + user_name + "'"
    # project_list = sql_engine(query, return_data=True) 
    project_list = psql.read_sql_query(query, sql_engine())
    print("--- Current Directory Read Time: %s seconds ---" % (time.time() - start_time))
    # print(project_list)
    return project_list


#create new problem statement
def upload_projects(project_id, project_name, problem_type, user_name):
    start_time = time.time()
    query = f"UPDATE public.base_data SET current_project='No' where user_name='{user_name}'".format(user_name)
    sql_engine().execute(f"UPDATE public.base_data SET current_project='No' where user_name='{user_name}'".format(user_name))
    # sql_engine()().execute(query)
    query=f"INSERT INTO public.base_data(project_id, project_name, user_name,problem_type,current_project) VALUES ('{str(project_id)}','{project_name}','{user_name}','{problem_type}','Yes')"
    # sql_engine()().execute(query)
    sql_engine().execute(query)

#For flaging the project in use
def update_projects(project_id, user_name):
    start_time = time.time()
    query = f"UPDATE public.base_data SET current_project='No' where user_name='{user_name}'".format(user_name)
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    query = f"UPDATE public.base_data SET current_project='Yes' where user_name='{user_name}'" \
            f" AND project_id='{project_id}'".format(user_name, project_id)
    sql_engine().execute(query)
    # sql_engine()().execute(query)

def delete_projects(project_id, user_name):
    start_time = time.time()

    query = f"DELETE from public.base_data where user_name='{user_name}' AND project_id='{project_id}'".\
    format(user_name, project_id)
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    query = f"DELETE from public.train_data where user_name='{user_name}' AND project_id='{project_id}'".\
    format(user_name, project_id)
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    query = f"DELETE from public.test_data where user_name='{user_name}' AND project_id='{project_id}'".\
    format(user_name, project_id)
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    query = f"DELETE from public.modelling where user_name='{user_name}' AND project_id='{project_id}'".\
    format(user_name, project_id)
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    query = f"DELETE from public.eda where user_name='{user_name}' AND project_id='{project_id}'".\
    format(user_name, project_id)
    # sql_engine()().execute(query)
    sql_engine().execute(query)

def retrieve_temp_operation():
    feat_engine = pd.read_pickle(os.path.join(temp_dir, 'temp_operations.pkl'))
    return feat_engine.loc[:, ['Operation', 'Column','Key']]

def use_operation(user_name):
    start_time = time.time()
    project_id = get_current_project(user_name)
    query = f"select operations from public.train_data where user_name='{user_name}' AND project_id='{project_id}' AND use_file='Yes'".format(user_name, project_id)
    print(query)
    data = psql.read_sql_query(query, sql_engine()).reset_index(drop=True)
    # data = sql_engine()(query, return_data=True).reset_index(drop=True)
    print('Stored Operations')
    operations=pickle.loads(data['operations'][0])
    print(operations)
    print("--- Problem Type Query Time: %s seconds ---" % (time.time() - start_time))
    return operations

def get_all_operations(user_name):
    start_time = time.time()
    project_id = get_current_project(user_name)
    query = f"select operations from public.train_data where user_name='{user_name}' AND project_id='{project_id}' AND use_file='Yes'".format(user_name, project_id)
    print(query)
    data = psql.read_sql_query(query, sql_engine()).reset_index(drop=True)
    # data = sql_engine()(query, return_data=True).reset_index(drop=True)
    print('Stored Operations')
    operations=pickle.loads(data['operations'][0])
    print(operations)
    print("--- Problem Type Query Time: %s seconds ---" % (time.time() - start_time))
    return operations

def get_current_project(user_name):
    query=f"select project_id from public.base_data where user_name='{user_name}' AND current_project='Yes'"
    # check = sql_engine()(query, return_data=True).reset_index(drop="True")
    check = pd.read_sql_query(f"select project_id from public.base_data where "
                              f"user_name='{user_name}' AND current_project='Yes'", sql_engine()).reset_index(drop="True")
    return check['project_id'][0]

def compress_df(df):
    print(df.info(verbose=False))
    for column in df:
        if df[column].dtype == 'float64':
            df[column]=pd.to_numeric(df[column], downcast='float')
        if df[column].dtype == 'int64':
            df[column]=pd.to_numeric(df[column], downcast='integer')
    print(df.info(verbose=False))
    return df

chart_options=['distplot', 'countplot','scatterplot','lineplot','boxplot','piechart','correlation']
charts=pd.DataFrame(columns=['chart','X','Y','Z'])
charts['chart']=chart_options

def new_file_queries(user_name, project_id, file_name, df):

    query = f"INSERT INTO public.train_data(user_name,project_id,file_name, data, display_data) " \
            f"VALUES ('{user_name}','{project_id}','{file_name}','{df.to_json()}', '{df.head(20).to_json()}') "
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    operations=pickle.dumps(pd.DataFrame(index=range(0),columns=['Operation', 'Column', 'Key']))
    sql_engine().execute("UPDATE public.train_data SET operations= %s where user_name=%s AND project_id=%s AND file_name=%s", operations,user_name,project_id,file_name)
    # sql_engine()().execute()
    # sql_engine().execute(query)
    query = f"INSERT INTO public.modelling(user_name,project_id,file_name) " \
            f"VALUES ('{user_name}','{project_id}','{file_name}') "
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    models=pickle.dumps(pd.DataFrame(index=range(0),columns=['model_name', 'model_file','fit_time']))
    sql_engine().execute("UPDATE public.modelling SET models= %s where user_name=%s AND project_id=%s AND file_name=%s", models,user_name,project_id,file_name)
    # sql_engine().execute(query)
    # sql_engine()().execute()    

    query = f"INSERT INTO public.test_data(user_name,project_id,file_name) " \
            f"VALUES ('{user_name}','{project_id}','{file_name}') "
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    query = f"INSERT INTO public.eda(user_name,project_id,file_name,charts) " \
            f"VALUES ('{user_name}','{project_id}','{file_name}','{charts.to_json()}') "
    # sql_engine()().execute(query)
    sql_engine().execute(query)

def file_upload(user_name, df,project_id,file_name):
    query = f"select file_name from public.train_data where user_name='{user_name}' AND project_id='{project_id}'".format(user_name, project_id)
    df=compress_df(df)
    start_time=time.time()
    base = psql.read_sql_query(query, sql_engine())
    # base = sql_engine()(query, return_data=True)
    if file_name not in base['file_name'].values:
        print('New File')
        new_file_queries(user_name, project_id, file_name, df)
    else:
        query = f"UPDATE public.train_data SET data='" + df.to_json() + \
                f"' WHERE user_name='{user_name}' AND project_id='{project_id}'".format(user_name,project_id)
        # sql_engine()().execute(query)
        sql_engine().execute(query)
    print("--- Upload Time: %s seconds ---" % (time.time() - start_time))


def merge_file_upload(user_name, df, file_name):
    
    print('File Name: ', file_name)
    start_time = time.time()
    project_id = get_current_project(user_name)
    print('Project Id: ',project_id)
    print('New File')
    df=compress_df(df)
    new_file_queries(user_name, project_id, file_name, df)

def current_file(user_name):
    
    start_time = time.time()
    project_id = get_current_project(user_name)
    #print(user_name)
    query = f"select file_name from public.train_data where user_name='{user_name}' AND project_id='{project_id}' and use_file='Yes'".format(user_name, project_id)
    data = psql.read_sql_query(query, sql_engine()).reset_index(drop=True)
    # data = sql_engine()(query, return_data=True).reset_index(drop=True)
    print("--- Database Read Time: %s seconds ---" % (time.time() - start_time))
    file_name = data['file_name'][0]
    print('File Name: ', file_name)
    return file_name

def get_current_directory(user_name):
    start_time = time.time()
    project_id = get_current_project(user_name)
    query = f"select file_name from public.train_data where user_name='{user_name}' AND project_id='{project_id}'".format(
        user_name, project_id)
    data = psql.read_sql_query(query, sql_engine())
    # data = sql_engine()(query, return_data=True)
    print("--- Current Directory Read Time: %s seconds ---" % (time.time() - start_time))
    return data


def delete_file(user_name, file_name):
    start_time = time.time()
    project_id = get_current_project(user_name)
    query = f"DELETE from public.train_data where user_name='{user_name}' AND project_id='{project_id}'" \
            f"AND file_name='{file_name}'".format(user_name, project_id, file_name)
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    query = f"DELETE from public.modelling where user_name='{user_name}' AND project_id='{project_id}'" \
            f"AND file_name='{file_name}'".format(user_name, project_id, file_name)
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    query = f"DELETE from public.test_data where user_name='{user_name}' AND project_id='{project_id}'" \
            f"AND file_name='{file_name}'".format(user_name, project_id, file_name)
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    query = f"DELETE from public.eda where user_name='{user_name}' AND project_id='{project_id}'" \
            f"AND file_name='{file_name}'".format(user_name, project_id, file_name)
    # sql_engine()().execute(query)
    sql_engine().execute(query)

    print("--- File Delete Time: %s seconds ---" % (time.time() - start_time))


def get_current_file(user_name, file_name):
    start_time = time.time()
    project_id = get_current_project(user_name)
    #print(user_name)
    query = f"select data from public.train_data where user_name='{user_name}' AND project_id='{project_id}'" \
            f"AND file_name='{file_name}'".format(user_name, project_id, file_name)
    data = psql.read_sql_query(query, sql_engine()).reset_index(drop=True)
    # data = sql_engine()(query, return_data=True).reset_index(drop=True)
    print("--- Database Read Time: %s seconds ---" % (time.time() - start_time))
    print('File Name: ', file_name)
    df=compress_df(pd.read_json(data['data'][0]))
    return df

def get_current_display_file(user_name, file_name):
    start_time = time.time()
    project_id = get_current_project(user_name)
    #print(user_name)
    query = f"select display_data from public.train_data where user_name='{user_name}' AND project_id='{project_id}'" \
            f"AND file_name='{file_name}'".format(user_name, project_id, file_name)
    data = psql.read_sql_query(query, sql_engine()).reset_index(drop=True)
    # data = sql_engine()(query, return_data=True).reset_index(drop=True)
    print("--- Database Read Time: %s seconds ---" % (time.time() - start_time))
    print('File Name: ', file_name)
    df=compress_df(pd.read_json(data['display_data'][0]))
    return df


def update_train_file(user_name, file_name, df):
    start_time = time.time()
    project_id = get_current_project(user_name)
    df=compress_df(df)
    print('File Name: ', file_name)
    query = "UPDATE public.train_data SET data='" + df.to_json() + f"' ,display_data='" + df.head(20).to_json() + f"' where user_name='{user_name}' AND project_id='{project_id}'" \
            f"AND file_name='{file_name}'".format(user_name, project_id, file_name)
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    print("--- Database Read Time: %s seconds ---" % (time.time() - start_time))


def test_file_upload(user_name, file_in_use, df):
    start_time = time.time()
    #file_name = str(x)
    project_id=get_current_project(user_name)
    df=compress_df(df)
    print(df.info(verbose=False))
    file_name=current_file()
    query = f"select file_name from public.test_data where user_name='{user_name}' AND project_id='{project_id}'".format(
        user_name, project_id)
    base = psql.read_sql_query(query, sql_engine())
    # base = sql_engine()(query, return_data=True)
    if file_name not in base['file_name'].values:
        print('New File')
        query = f"INSERT INTO public.test_data(user_name,project_id,file_name, test_data) " \
            f"VALUES ('{user_name}','{project_id}','{file_name}','{df.to_json()}') "
        # sql_engine()().execute(query)
        sql_engine().execute(query)
    else:
        query = f"UPDATE public.test_data SET test_data='" + df.to_json() + \
                f"' WHERE user_name='{user_name}' AND project_id='{project_id}' AND file_name='{file_name}'".format(user_name,project_id,file_name)
        # sql_engine()().execute(query)
        sql_engine().execute(query)
    print("--- Test Data Upload Time: %s seconds ---" % (time.time() - start_time))


def get_test_file(user_name, file_name):
    start_time = time.time()
    project_id=get_current_project(user_name)
    query = f"select * from public.test_data where user_name='{user_name}' AND project_id='{project_id}'" \
            f"AND file_name='{file_name}'".format(user_name, project_id, file_name)
    data = psql.read_sql_query(query, sql_engine())
    # data = sql_engine()(query, return_data=True)
    print("--- Database Read Time: %s seconds ---" % (time.time() - start_time))
    print('File Name: ', file_name)
    df=compress_df(pd.read_json(data['test_data'][0]))
    return df


def update_test_file(user_name, file_name, df):
    start_time = time.time()
    project_id=get_current_project(user_name)
    print(df.info(verbose=False))
    df=compress_df(df)
    print(df.info(verbose=False))
    query = "UPDATE public.test_data SET test_data='" + df.to_json() + f"' where user_name='{user_name}' AND project_id='{project_id}'" \
            f"AND file_name='{file_name}'".format(user_name, project_id, file_name)
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    print("--- Database Read Time: %s seconds ---" % (time.time() - start_time))
    print('File Name: ', file_name)


def get_problem_type(user_name, file_name):
    start_time = time.time()
    project_id = get_current_project(user_name)
    query = f"select problem_type from public.base_data where user_name='{user_name}' AND project_id='{project_id}'".format(user_name, project_id)
    #query = "select problem_type from mlgui where file_name='" + str(file_name) + "'"
    data = psql.read_sql_query(query, sql_engine())
    # data = sql_engine()(query, return_data=True)
    data = data.reset_index(drop=True)
    print("--- Problem Type Query Time: %s seconds ---" % (time.time() - start_time))
    return data['problem_type'][0]


def save_models(user_name, current_file, models):
    start_time = time.time()
    project_id = get_current_project(user_name)
    query = f"UPDATE public.modelling SET models='" + models + f"' where user_name='{user_name}' AND project_id='{project_id}'" \
            f"AND file_name='{current_file}'".format(target_column, user_name, project_id, current_file)
    # = "UPDATE public.mlgui SET models='" + models.to_json() + "' WHERE file_name='" + file_name + "'"
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    print("--- Model Save Time: %s seconds ---" % (time.time() - start_time))


def get_models(user_name, file_name):
    start_time = time.time()
    project_id = get_current_project(user_name)

    query=f"select models from public.modelling where user_name='{user_name}' AND project_id='{project_id}' AND file_name='{file_name}'".format(user_name,project_id,file_name)
    try:
        # saved_models=compress_pickle.loads(sql_engine()(query, return_data=True).reset_index(drop="True")['models'][0], compression='gzip')
        saved_models=compress_pickle.loads(pd.read_sql_query(query, sql_engine()).reset_index(drop="True")['models'][0], compression='gzip')
    except:
        # saved_models=pickle.loads(sql_engine()(query, return_data=True).reset_index(drop="True")['models'][0])
        saved_models=pickle.loads(pd.read_sql_query(query, sql_engine()).reset_index(drop="True")['models'][0])
    #saved_models=pickle.loads(pd.read_sql_query(query, engine).reset_index(drop="True")['models'][0])
    print("--- Database Read Time: %s seconds ---" % (time.time() - start_time))
    print('File Name: ', file_name)
    return saved_models


def save_predictions_file(user_name, file_name, df):
    start_time = time.time()
    #query = f"UPDATE public.mlgui SET predictions='" + df.to_json() + "' WHERE file_name='" + file_in_use + "'"
    project_id = get_current_project(user_name)
    query = f"UPDATE public.test_data SET predictions= '" + df.to_json() + f"' where user_name='{user_name}' AND project_id='{project_id}'" \
                                                                            f"AND file_name='{file_name}'".format(user_name, project_id, file_name)

    # sql_engine()().execute(query)
    sql_engine().execute(query)
    print("--- Problem Type Query Time: %s seconds ---" % (time.time() - start_time))


def get_predictions_file(user_name, file_name):
    start_time = time.time()
    project_id = get_current_project(user_name)
    query=f"select predictions from public.test_data where user_name='{user_name}' AND project_id='{project_id}' AND file_name='{file_name}'".format(user_name,project_id,file_name)

    data = psql.read_sql_query(query, sql_engine())
    # data = sql_engine()(query, return_data=True)
    data = data.reset_index(drop=True)
    print("--- Problem Type Query Time: %s seconds ---" % (time.time() - start_time))
    return pd.read_json(data['predictions'][0]).to_csv()


def save_target(user_name, current_file, target_column):
    start_time = time.time()
    print('Target: ', target_column)
    project_id = get_current_project(user_name)
    query = f"UPDATE public.train_data SET target= '{target_column}' where user_name='{user_name}' AND project_id='{project_id}'" \
            f"AND file_name='{current_file}'".format(target_column, user_name, project_id, current_file)
    #query = "UPDATE public.mlgui SET target='" + target_column + "' WHERE file_name='" + current_file + "'"
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    print("--- Target Update Time: %s seconds ---" % (time.time() - start_time))


def get_target(user_name, file_name,project_id):
    start_time = time.time()
    query=f"select target from public.train_data where user_name='{user_name}' AND project_id='{project_id}' AND file_name='{file_name}'".format(user_name,project_id,file_name)

    data = psql.read_sql_query(query, sql_engine()).reset_index(drop=True)
    # data = sql_engine()(query, return_data=True).reset_index(drop=True)
    print("--- Target Acquire Time: %s seconds ---" % (time.time() - start_time))
    print('Target: ', data['target'][0])
    return data['target'][0]

def save_columns_selected(user_name, current_file, columns):
    start_time = time.time()
    project_id = get_current_project(user_name)
    query = f"UPDATE public.train_data SET columns_selected= '" + columns + f"' where user_name='{user_name}' AND project_id='{project_id}'" \
            f"AND file_name='{current_file}'".format(user_name, project_id, current_file)
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    print("--- Columns Update Time: %s seconds ---" % (time.time() - start_time))


def get_columns_selected(user_name, current_file):
    print('Current file: ', current_file)
    start_time = time.time()
    project_id = get_current_project(user_name)
    query = f"select columns_selected from public.train_data where user_name='{user_name}' AND project_id='{project_id}'" \
            f" AND file_name='{current_file}'".format(user_name, project_id, current_file)
    print('Query: ', query)
    data = psql.read_sql_query(query, sql_engine()).reset_index(drop=True)
    # data = sql_engine()(query, return_data=True).reset_index(drop=True)
    print("--- Getting Selected Columns Time: %s seconds ---" % (time.time() - start_time))
    return list(pd.read_json(data['columns_selected'][0]).iloc[:, 0].values)


def save_leaderboard(user_name, current_file, leaderboard):
    start_time = time.time()
    project_id = get_current_project(user_name)
    query = f"UPDATE public.modelling SET leaderboard= '" + leaderboard.to_json() + f"' where user_name='{user_name}' AND project_id='{project_id}'" \
                                                                            f"AND file_name='{current_file}'".format(
        user_name, project_id, current_file)
    # sql_engine()().execute(query)
    sql_engine().execute(query)
    print("--- Saving Leaderboard Time: %s seconds ---" % (time.time() - start_time))


def get_leaderboard(user_name, current_file, problem_type):
    class_columns = ['Model', 'Accuracy', 'Precision ', 'Recall ', 'F1 Score ', 'ROC Score ']
    reg_columns = ['Model', 'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error', 'R2 Score ']
    prob_type = {'classification': class_columns, 'regression': reg_columns}
    start_time = time.time()
    project_id = get_current_project(user_name)
    query = f"select leaderboard from public.modelling where user_name='{user_name}' AND project_id='{project_id}' AND file_name='{current_file}'".format(
        user_name, project_id,current_file)
    # data = sql_engine()(query, return_data=True)
    data = psql.read_sql_query(query, sql_engine())
    try:
        result = pd.read_json(data['leaderboard'][0])
    except ValueError as error:
        result = pd.DataFrame(columns=prob_type.get(problem_type))
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    print(result)
    return result


def store_all_operation(user_name, current_file, operations):
    project_id = get_current_project(user_name)
    operations=pickle.dumps(operations)
    sql_engine().execute("UPDATE public.train_data SET operations= %s where user_name=%s AND project_id=%s AND use_file='Yes'",operations,user_name,project_id)

def temporary(user_name, temp_data):
    project_id=get_current_project(user_name)
    query=f"select temp from public.train_data where user_name='{user_name}' AND project_id='{project_id}' AND use_file='Yes'".format(user_name, project_id)
    start_time = time.time()

    data=pd.read_sql_query(query, sql_engine())
    # data=sql_engine(query, return_data=True)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    if data['temp'][0]==None:
        query = f"UPDATE public.train_data SET temp='" + temp_data.to_json() + \
                f"' WHERE user_name='{user_name}' AND project_id='{project_id}'".format(user_name,project_id)
        # sql_engine()().execute(query)
        sql_engine().execute(query)
        print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    else:

        df=pd.read_json(data['temp'][0])
        feats=list(set(temp_data.columns)-set(df.columns))
        temp_data=temp_data.loc[:,feats]
        data=pd.concat([df,temp_data],axis=1)
        query = f"UPDATE public.train_data SET temp='" + data.to_json() + \
                f"' WHERE user_name='{user_name}' AND project_id='{project_id}'".format(user_name,project_id)
        # sql_engine().execute(query)
        sql_engine().execute(query)
        print("--- Fit Time: %s seconds ---" % (time.time() - start_time))

def temp_data_remove(user_name, list1):
    project_id=get_current_project(user_name)
    query=f"select temp from public.train_data where user_name='{user_name}' AND project_id='{project_id}' AND use_file='Yes'".format(user_name, project_id)
    start_time = time.time()

    data=pd.read_sql_query(query, sql_engine())
    # data=sql_engine(query, return_data=True)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    df=pd.read_json(data['temp'][0])
    df.drop(list1,axis=1,inplace=True)
    query = f"UPDATE public.train_data SET temp='" + df.to_json() + \
            f"' WHERE user_name='{user_name}' AND project_id='{project_id}' AND use_file='Yes'".format(user_name,project_id)
    # sql_engine().execute(query)
    sql_engine().execute(query)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))