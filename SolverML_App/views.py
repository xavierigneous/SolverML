from django.shortcuts import render, HttpResponse, redirect
from django.http import HttpResponse, FileResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from sklearn import preprocessing
from sklearn.exceptions import NotFittedError
import pickle, time, sqlalchemy, autofeat
import compress_pickle
from sqlalchemy.exc import SQLAlchemyError
import matplotlib.pyplot as plt
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')
import scikitplot.metrics
import scipy.stats as stats
import plotly.express as px
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest, RFECV
# import cx_Oracle
import seaborn as sns
sns.set()
sns.color_palette("cubehelix")
from functools import reduce
import json, base64
import shap, lime, eli5
from eli5.sklearn import PermutationImportance
from shap.plots._force_matplotlib import draw_additive_plot
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
import pandas.io.sql as psql
from pandas.api.types import is_string_dtype,is_numeric_dtype
from collections import defaultdict
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
# from IPython.display import HTML
# import AppConfig
import os, sqlalchemy, random, sys
from sshtunnel import SSHTunnelForwarder
from psycopg2.extensions import register_adapter, AsIs
register_adapter(np.int64, AsIs)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier, \
    AdaBoostClassifier, AdaBoostRegressor, RandomForestRegressor
import xgboost
from xgboost import XGBClassifier, XGBRegressor
from sklearn import model_selection, tree, preprocessing, metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, LeaveOneOut, ShuffleSplit, RepeatedKFold
from distributed import Client
import graphviz, joblib, dask_ml, datetime
import dask_ml.model_selection as dcv
from sklearn.tree import export_text, export_graphviz, plot_tree
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import psycopg2 as pg
import warnings
warnings.filterwarnings('ignore')

################# DB Operations import ###########################
abspath = os.path.abspath(__file__)
working_directory = os.path.dirname(abspath)
os.chdir(working_directory)
sys.path.append(working_directory)
from db_operations import sql_engine, get_projects, upload_projects, update_projects, delete_projects, get_current_project, \
    use_operation, get_all_operations, new_file_queries, file_upload, merge_file_upload, current_file, get_current_directory, \
    delete_file, get_current_file, get_current_display_file, update_train_file, test_file_upload, get_test_file, update_test_file, \
    get_problem_type, save_models, get_models, save_predictions_file, get_predictions_file, save_target, get_target, \
    save_columns_selected, get_columns_selected, save_leaderboard, get_leaderboard, store_all_operation, store_chart, temporary, \
    retrieve_temp_operation, temp_data_remove

############### PLotting Operations import #######################
from metrics_plot import plot, plot_classification, plot_regression, linear_feat_importance, tree_feat_importance, feat_importance_init
############## Modelling Operations import #######################
from modelling_utilities import fit_ml_algo, fit_ml_algo_predict, roc_auc_score_multiclass, train_test_fit
################# Tuning Operations import #######################
from tuning_ops import classify_metric, \
    rf_regress, xgb_regress, log_regress, knn_regress, svc_regress, dtree_regress, adaboost_regress, gradboost_regress, \
    rf_classify, xgb_classify, log_classify, knn_classify, svc_classify, dtree_classify, adaboost_classify, gradboost_classify, \
    class_split, reg_split
class_cross_valid=['KFold','Stratified Kfold']
reg_cross_valid=['Kfold','Shuffle Split']

# Create your views here.
global nrows, temp_file, user_name
temp_dir = os.path.join(os.getcwd(), 'temp_data')
print(temp_dir)
import getpass
# user_name = getpass.getuser()
# print('Welcome, ', user_name)

def SolverML(request):
    return render(request, 'SolverML.html')

def login_page(request):
    if 'login' in request.POST and request.method == 'POST':
        # Process the request if posted data are available
        username = request.POST['username']
        password = request.POST['password']
        # Check username and password combination if correct
        user = authenticate(username=username, password=password)
        if user is not None:
            # Save session as cookie to login the user
            login(request, user)
            # Success, now let's login the user.
            request.session['user_id'] = user.username
            return redirect('homepage')
        
        else:
            # Incorrect credentials, let's throw an error to the screen.
            return render(request, 'login.html', {'error_message': 'Incorrect username and / or password.'})
    if 'logout' in request.POST and request.method == 'POST':
            logout(request)
            return redirect('login_page')
    
    else:
        # No post data availabe, let's just show the page to the user.
        return render(request, 'login.html')


def id_generator(lst):
    return sorted(set(range(1, lst[-1])) - set(lst))
def home_context(user_name):
    projects = get_projects(user_name).sort_values(by='project_id',ascending=True)
    projects=projects[['project_id','project_name','problem_type']]
    projects = projects.to_dict(orient='records')
    return projects

def home(request):
    user_name = request.session['user_id']
    try:
        current_project = get_current_project(user_name)
    except:
        current_project=''
    projects = home_context(user_name)
    
    if 'project_submit' in request.POST and request.method == "POST":
        project_name = request.POST.get('problem_name')
        problem_type = request.POST['problem_state_type']
        projects = get_projects(user_name).sort_values(by='project_id')
        try:
            ids=list(projects.groupby('user_name').get_group(user_name)['project_id'].sort_values().values)
            if len(id_generator(ids))==0 and len(ids)>0==0:
                project_id=np.array(ids).max()+1
            elif len(id_generator(ids))==0 and len(ids)==0:
                project_id = 1
            else:
                project_id = id_generator(ids)[0]
        except:
            project_id = 1
        upload_projects(project_id, project_name, problem_type, user_name)
        projects = home_context(user_name)
        return render(request, 'homepage.html', {'projects': projects})
    if 'select_project' in request.POST and request.method == "POST":
        print('Project ID: ', request.POST.get('select_project'))
        project_id = request.POST.get('select_project')
        update_projects(project_id, user_name)
        projects = home_context(user_name)
        return redirect('datainput')
        # return render(request, 'homepage.html', {'projects': projects})

    if 'delete_project' in request.POST and request.method == "POST":
        print('Project ID: ', request.POST.get('delete_project'))
        project_id = request.POST.get('delete_project')
        delete_projects(project_id, user_name)
        projects = home_context(user_name)
        return render(request, 'homepage.html', {'projects': projects})
    if 'logout' in request.POST and request.method == 'POST':
            logout(request)
            return redirect('login_page')
    else:
        return render(request, 'homepage.html', {'projects': projects, 'current_project': current_project})

file_type = {'csv': pd.read_csv, 'xlsx': pd.read_excel, 'xls': pd.read_excel, 'txt': pd.read_table}

def datainput(request):
    user_name = request.session['user_id']
    if 'csv_upload' in request.POST and request.method == "POST":
        # file=request.FILES['file']
        print(request.POST)
        print(request.FILES.getlist("file"))
        for x in request.FILES.getlist("file"):
            file_name = str(x)
            print('File Name: ', file_name)
            start_time = time.time()
            project_id = get_current_project(user_name)
            print('Project Id: ',project_id)
            df = file_type.get(file_name.split('.')[-1])(x, na_values=' ')
            file_upload(user_name, df,project_id,file_name)

        project_id = get_current_project(user_name)
        query=f"select project_name from public.base_data where user_name='{user_name}' AND project_id='{project_id}'".format(user_name,project_id)
        project_name=pd.read_sql_query(query, sql_engine()).reset_index(drop="True")['project_name'][0]
        # project_name=sql_engine(query, return_data=True).reset_index(drop="True")['project_name'][0]
        directory = get_current_directory(user_name)
        hr = directory['file_name']
        file_lists = [{'field': f, 'title': f} for f in hr]
        upload_view = {'file_lists': file_lists,'project_name':project_name}
        messages.success(request, 'File Upload Successful',extra_tags='csv_upload')
        return render(request, "datainput.html", upload_view)

    if 'url_upload' in request.POST and request.method == "POST":

        print(request.POST.getlist("url"))
        print(request.POST.get("url_file_name"))
        for x in request.POST.getlist("url"):
            file_name = request.POST.get("url_file_name")
            print('File Name: ', file_name)
            start_time = time.time()
            project_id = get_current_project(user_name)
            print('Project Id: ',project_id)
            df = pd.read_csv(x, na_values=' ')
            print(df)
            file_upload(user_name, df,project_id,file_name)

        project_id = get_current_project(user_name)
        query=f"select project_name from public.base_data where user_name='{user_name}' AND project_id='{project_id}'".format(user_name,project_id)
        project_name=pd.read_sql_query(query, sql_engine()).reset_index(drop="True")['project_name'][0]
        # project_name=sql_engine(query, return_data=True).reset_index(drop="True")['project_name'][0]
        directory = get_current_directory(user_name)
        hr = directory['file_name']
        file_lists = [{'field': f, 'title': f} for f in hr]
        upload_view = {'file_lists': file_lists,'project_name':project_name}
        messages.success(request, 'File Upload Successful',extra_tags='url_upload')
        return render(request, "datainput.html", upload_view)

    if 'sql_upload' in request.POST and request.method=="POST":
        project_id = get_current_project(user_name)
        hostname=request.POST.get('Hostname')
        port=str(request.POST.get('port'))
        username=str(request.POST.get('username'))
        pwd=str(request.POST.get('password'))
        database=str(request.POST.get('database'))
        table=str(request.POST.get('table'))
        query=str(request.POST.get('query'))
        print(hostname,port,username,pwd,database,query)
        print(query)
        print('Engine: ')
        print('postgres://'+username+':'+pwd+'@'+hostname+':'+port+'/'+database)
        eng=sqlalchemy.create_engine('postgresql://'+username+':'+pwd+'@'+hostname+':'+port+'/'+database)
        #engine = sqlalchemy.create_engine('postgresql://analytics:analytics@203.112.158.89:5432/ratings')
        if query=="":
            data = psql.read_sql_table(table,eng)
        else:
            data=psql.read_sql_query(query,eng)
        file_upload(data,project_id,table)
        print(data)


    if 'merge' in request.POST and request.method == "POST":
        # print('Requests: ',request.POST)
        print('File Selected ', request.POST.getlist('choose_file'))
        cho_file = request.POST.getlist('choose_file')
        print(cho_file)
        project_id = get_current_project(user_name)
        data_in1 = get_current_file(cho_file[0])
        data_in2 = get_current_file(cho_file[1])
        columns1 = [{'field': f, 'title': f} for f in data_in1.columns.to_list()]
        columns2 = [{'field': f, 'title': f} for f in data_in2.columns.to_list()]
        query = f"INSERT INTO public.merge_table(user_name,project_id,filename1, filename2) " \
            f"VALUES ('{user_name}','{project_id}','{cho_file[0]}','{cho_file[1]}') "
        sql_engine().execute(query)
        directory = get_current_directory(user_name)
        hr = directory['file_name']
        file_lists = [{'field': f, 'title': f} for f in hr]
        query=f"select project_name from public.base_data where user_name='{user_name}' AND project_id='{project_id}'".format(user_name,project_id)
        project_name=pd.read_sql_query(query, sql_engine()).reset_index(drop="True")['project_name'][0]
        # project_name=sql_engine(query, return_data=True).reset_index(drop="True")['project_name'][0]
        data_object = {
            'columns1': columns1,
            'columns2': columns2,
            'file_lists': file_lists
        }

        return render(request, 'datainput.html', data_object)
    if 'merge_sub' in request.POST and request.method == "POST":
        # print('Requests: ',request.POST)
        print('File Selected ', request.POST.getlist('choose_file'))
        #cho_file = request.POST.getlist('choose_file')
        # print(cho_file)
        project_id = get_current_project(user_name)
        query=f"SELECT filename1,filename2, user_name, project_id FROM merge_table where user_name='{user_name}' AND project_id='{project_id}'".format(user_name,project_id)
        data=sql_engine()(query, return_data=True).reset_index(drop=True)
        file1, file2 = data['filename1'][0], data['filename2'][0]
        data_in1 = get_current_file(file1)
        data_in2 = get_current_file(file2)
        columns1 = [{'field': f, 'title': f} for f in data_in1.columns.to_list()]
        columns2 = [{'field': f, 'title': f} for f in data_in2.columns.to_list()]
        type_join = request.POST['type_join']
        col1 = request.POST['columns1']
        col2 = request.POST['columns2']

        print('Columns Chosen ', col1, type_join, col2)

        data_in = pd.merge(data_in1, data_in2, how=type_join, on=[col1, col2])
        print('Dataframe Shape: ', data_in.shape)
        #data_in.to_csv(os.path.join(temp_dir, 'Merge.csv'), index=False)
        file_name=str(request.POST.get('merge_name'))
        merge_file_upload(data_in, file_name)
        query = f"UPDATE public.merge_table SET column1 = '{col1}', column2 = '{col2}', type_join = '{type_join}', file_name = '{file_name}' where user_name='{user_name}' and project_id='{project_id}' and filename1='{file1}' and filename2='{file2}'".format(col1, col2,type_join,file_name,user_name,project_id,file1,file2)
        # engine.execute()
        sql_engine().execute(query)

        query = f"UPDATE public.train_data SET use_file='Yes' where user_name='{user_name}' and project_id='{project_id}' and file_name='{file_name}'".format(user_name,project_id,file_name)
        # engine.execute()
        sql_engine().execute(query)
        directory = get_current_directory(user_name)
        hr = directory['file_name']
        file_lists = [{'field': f, 'title': f} for f in hr]
        query=f"select project_name from public.base_data where user_name='{user_name}' AND project_id='{project_id}'".format(user_name,project_id)
        project_name=pd.read_sql_query(query, sql_engine()).reset_index(drop="True")['project_name'][0]
        # project_name=sql_engine(query, return_data=True).reset_index(drop="True")['project_name'][0]
        data_object = {
            'columns1': columns1,
            'columns2': columns2,
            'data_object': data_in.head(20).to_html(index=False, index_names=False),
            'file_lists': file_lists
        }

        return render(request, 'datainput.html', data_object)


    if 'use_file' in request.POST and request.method == "POST":
        print('Using file ', request.POST['use_file'])
        file_name=request.POST['use_file']
        project_id = get_current_project(user_name)
        query = f"UPDATE public.train_data SET use_file='No' where user_name='{user_name}' and project_id='{project_id}'".format(user_name,project_id)
        # engine.execute()
        sql_engine().execute(query)
        query = f"UPDATE public.train_data SET use_file='Yes' where user_name='{user_name}' and project_id='{project_id}' and file_name='{file_name}'".format(user_name,project_id,file_name)
        # engine.execute()
        sql_engine().execute(query)

        directory = get_current_directory(user_name)
        hr = directory['file_name']
        print('File list:')
        print(hr)
        file_lists = [{'field': f, 'title': f} for f in hr]
        query=f"select project_name from public.base_data where user_name='{user_name}' AND project_id='{project_id}'".format(user_name,project_id)
        project_name=pd.read_sql_query(query, sql_engine()).reset_index(drop="True")['project_name'][0]
        # project_name=sql_engine(query, return_data=True).reset_index(drop="True")['project_name'][0]
        return render(request, "datainput.html", {'file_lists': file_lists,'project_name':project_name})

    if 'delete' in request.POST and request.method == "POST":
        # print('Requests: ',request.POST)
        print('File Selected ', request.POST.getlist('choose_file'))
        cho_file = request.POST.getlist('choose_file')
        print(cho_file)

        for file in cho_file:
            delete_file(user_name, file)
        directory = get_current_directory(user_name)
        hr = directory['file_name']
        file_lists = [{'field': f, 'title': f} for f in hr]
        project_id = get_current_project(user_name)
        query=f"select project_name from public.base_data where user_name='{user_name}' AND project_id='{project_id}'".format(user_name,project_id)
        project_name=pd.read_sql_query(query, sql_engine()).reset_index(drop="True")['project_name'][0]
        # project_name=sql_engine(query, return_data=True).reset_index(drop="True")['project_name'][0]
        data_object = {

            'file_lists': file_lists
        }
        return render(request, "datainput.html", {'file_lists': file_lists,'project_name':project_name})

    else:
        project_id = get_current_project(user_name)
        query=f"select project_name from public.base_data where user_name='{user_name}' AND project_id='{project_id}'".format(user_name,project_id)
        project_name=pd.read_sql_query(query, sql_engine()).reset_index(drop="True")['project_name'][0]
        # project_name=sql_engine(query, return_data=True).reset_index(drop="True")['project_name'][0]
        directory = get_current_directory(user_name)
        hr = directory['file_name']
        file_lists = [{'field': f, 'title': f} for f in hr]
        upload_view = {'file_lists': file_lists,'project_name':project_name}
        return render(request, "datainput.html", upload_view)

def dataview_context(user_name, temp_file):
    data_in = get_current_display_file(user_name, temp_file)
    columns = [{'field': f, 'title': f} for f in data_in.columns.to_list()]
    project_id = get_current_project(user_name)
    query=f"select project_name from public.base_data where user_name='{user_name}' AND project_id='{project_id}'".format(user_name,project_id)
    project_name=pd.read_sql_query(query, sql_engine()).reset_index(drop="True")['project_name'][0]
    # project_name=sql_engine(query, return_data=True).reset_index(drop="True")['project_name'][0]
    return project_id, project_name, data_in, columns

def dataview(request):
    user_name = request.session['user_id']
    if 'view' in request.POST and request.method == "POST":
        # print('Requests: ',request.POST)
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        project_id, project_name, data_in, columns = dataview_context(user_name, temp_file)
        data_object = {
            'data_object': data_in.head(20).to_html(index=False, index_names=False)
            , 'columns': columns,
            'project_name':project_name
        }

        return render(request, 'dataview.html', data_object)

    if 'describe' in request.POST and request.method == "POST":
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        project_id, project_name, data_in, columns = dataview_context(user_name, temp_file)
        des = data_in.describe(include='all').T
        des['Data Type'] = data_in.dtypes
        des['Missing Observations'] = data_in.isna().sum()
        des = des.rename_axis('Columns').reset_index(inplace=False)
        data_described = des.to_html(index=False)

        return render(request, 'dataview.html', {'data_described': data_described, 'columns': columns,'project_name':project_name})

    if 'columnlist' in request.POST and request.method == "POST":
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        project_id, project_name, data_in, columns = dataview_context(user_name, temp_file)
        listcolumn = pd.DataFrame({'Uniques': data_in.apply(pd.Series.nunique)})
        listcolumn['Data Type'] = data_in.dtypes
        listcolumn['Missing Observations'] = data_in.isna().sum()
        listcolumn['Uniques'] = data_in.apply(pd.Series.nunique).to_list()
        listcolumn = listcolumn.rename_axis('Columns').reset_index(inplace=False)
        print(listcolumn)
        columnlist_described = listcolumn.to_html(index=False)
        
        return render(request, 'dataview.html', {'columnlist_described': columnlist_described, 'columns': columns,'project_name':project_name})

    if 'graph_view' in request.POST and request.method == "POST":

        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        project_id, project_name, data_in, columns = dataview_context(user_name, temp_file)
        plt.clf()
        filter_columns = [i for i in data_in.columns if
                          (data_in[i].dtypes in ['float64', 'int64', 'float32', 'int32'])] + [i for i in data_in.columns
                                                                                              if ((
                    data_in[i].dtypes in ['object', 'bool'])) and data_in[i].nunique() < 10]
        l = (len(filter_columns) // 3)  # +(len(data_in.columns)%3)
        fig, axes = plt.subplots(figsize=(l * 2, l * 2), nrows=l, ncols=3, squeeze=True)
        # fig, axes = plt.subplots( figsize=(16, 16),nrows=l, ncols=3,squeeze=True)
        for ax, col in zip(axes.flat, filter_columns):
            print('Axis No: ', ax)
            if data_in[col].dtypes in ['float64', 'int64', 'float32', 'int32']:
                print(col)
                sns.distplot(data_in[col], ax=ax)
            elif (data_in[col].dtypes == 'object' or 'bool') and (data_in[col].nunique() < 10):
                print(col)
                sns.countplot(data_in[col], ax=ax)
            plt.tight_layout()

        distribution = plot()
        
        data_object = {
            'data_object': data_in.head(20).to_html(index=False, index_names=False),
            'distribution': distribution,
            'columns': columns,
            'project_name':project_name
        }
        return render(request, 'dataview.html', data_object)

    if 'target' in request.POST and request.method == "POST":

        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        start_time = time.time()
        target = request.POST['columns']
        save_target(user_name, temp_file, target)
        project_id, project_name, data_in, columns = dataview_context(user_name, temp_file)
        return render(request, "dataview.html",
                      {'columns': columns, 'data_object': data_in.head(20).to_html(index=False, index_names=False),
                      'project_name':project_name})

    else:
        temp_file = current_file(user_name)
        print(temp_file)
        project_id, project_name, data_in, columns = dataview_context(user_name, temp_file)
        return render(request, "dataview.html", {'columns': columns,
                                                 'data_object': data_in.head(20).to_html(index=False,
                                                                                         index_names=False),
                                                 'project_name':project_name})


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


def transform(request):
    user_name = request.session['user_id']
    # print(request.POST)
    if 'submit_transformation' in request.POST and request.method == "POST":
        print(request.POST['select_transformation'])
        operation_chosen = request.POST['select_transformation']
        print("Operation Chosen: ", operation_chosen)
        if 'Label Encoding' in operation_chosen:
            # print('Requests: ',request.POST)
            temp_file = current_file(user_name)
            # data_in=pd.read_csv(temp_file)
            data_in = get_current_file(user_name, temp_file)
            print('Label Encoding in Progress')
            feats = request.POST.getlist('cols')
            print(data_in.loc[:, feats].head())
            le = preprocessing.LabelEncoder()
            d = defaultdict(preprocessing.LabelEncoder)
            # data_in.loc[:, feats] = data_in.loc[:, feats].apply(lambda x: d[x.name].fit_transform(x))
            # store_temp_operation(d,feats)
            op = d.default_factory.__name__
            key = [op + '_' + x for x in feats if not str(x) == "nan"]
            print(key)
            df=use_operation(user_name)
            if not key in df['Key'].to_list():
                print('New Operation')
                temp = pd.DataFrame(data_in.loc[:, feats])
                temp.loc[:, :] = temp.loc[:, feats].apply(lambda x: d[x.name].fit_transform(x))
                op = type(list(d.values())[0]).__name__
                temp = temp.loc[:, :].add_suffix('_' + str(op))
                print(temp)
                print(pd.concat([data_in, temp], axis=1))
                temp = pd.concat([data_in, temp], axis=1)
                store_temp_operation(d, feats)
            else:
                messages.error(request, 'Operation Already Performed')
                print('Operation already performed')
                temp = data_in.copy()


            columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
            transform_data = {
                'transform_data': temp.head(20).to_html(index=False, index_names=False)
                , 'columns': columns
            }
            # print('Print after view: ',request.POST.getlist('cols'))
            # data_in.to_csv(temp_file, index=False)
            return render(request, 'transform.html', transform_data)
        if 'One Hot Encoding' in operation_chosen:
            # print('Requests: ',request.POST)
            temp_file = current_file(user_name)
            # data_in=pd.read_csv(temp_file)
            data_in = get_current_file(user_name, temp_file)
            feats = request.POST.getlist('cols')
            print('One Hot Encoding in Progress on : ', feats)
            print(onehotencode(data_in, feats))
            print(onehotencode.__name__)
            # data_in=onehotencode(data_in,feats)
            # store_operation(onehotencode,feats)

            temp = pd.DataFrame(data_in.loc[:, feats])
            temp = onehotencode(temp, feats)
            # op=onehotencode.__name__
            # temp=temp.loc[:,:].add_suffix(op)
            print(temp)
            print(pd.concat([data_in, temp], axis=1))
            temp = pd.concat([data_in, temp], axis=1)
            store_temp_operation(onehotencode, feats)
            # update_train_file(temp_file, data_in)
            # data_in.to_csv(temp_file, index=False)
            columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
            transform_data = {
                'transform_data': temp.head(20).to_html(index=False, index_names=False)
                , 'columns': columns
            }
            return render(request, 'transform.html', transform_data)
        
        
        if 'Log Transform' in operation_chosen:
            # print('Requests: ',request.POST)
            temp_file = current_file(user_name)
            # data_in=pd.read_csv(temp_file)
            data_in = get_current_file(user_name, temp_file)
            
            feats = request.POST.getlist('cols')
            print('Log Transform on ',feats)
            # log_transform = Pipeline([('logtransform',
            #                         FunctionTransformer(log.log_transform, inverse_func=log.log_inverse_transform,
            #                                             check_inverse=True))])
            log_transform = Pipeline([('logtransform',
                                    FunctionTransformer(transformations.log_transform))])
            log_transform_pipeline = ColumnTransformer([
                ('_log', Pipeline([('log_transform', log_transform)]), feats)])
            
            temp = pd.DataFrame(data_in.loc[:, feats])
            if not temp.isna().any().any():
                print('Log Transform in Progress')
                op = log_transform_pipeline.transformers[0][0]
                key = [[op + '_' + x for x in feats if not str(x) == "nan"]]
                print(key)
                df=use_operation(user_name)
                if not key in df['Key'].to_list():
                    print('New Operation')
                    temp.loc[:, :] = log_transform_pipeline.fit_transform(temp)
                    temp = temp.loc[:, :].add_suffix(op)
                    print(temp)
                    #print(pd.concat([data_in, temp], axis=1))
                    store_temp_operation(log_transform_pipeline, feats)
                    temp = pd.concat([data_in, temp], axis=1)
                else:
                    messages.error(request, 'Operation Already Performed')
                    print('Operation already performed')
                    temp = data_in.copy()
            else:
                print('Not happening')
                messages.error(request, 'Null Values Found in ',feats)
                messages.error(request, feats)
                temp = data_in.copy()
            print(temp)
            # update_train_file(temp_file, data_in)
            columns = [{'field': f, 'title': temp[f].dtypes} for f in temp.columns.to_list()]
            transform_data = {
                'transform_data': temp.head(20).to_html(index=False, index_names=False),
                'columns': columns,
            }
            # data_in.to_csv(temp_file, index=False)
            return render(request, 'transform.html', transform_data)

        if 'Square Root Transform' in operation_chosen:
            # print('Requests: ',request.POST)
            temp_file = current_file(user_name)
            # data_in=pd.read_csv(temp_file)
            data_in = get_current_file(user_name, temp_file)
            
            feats = request.POST.getlist('cols')
            print('Square Root Transform on ',feats)
            # log_transform = Pipeline([('logtransform',
            #                         FunctionTransformer(log.log_transform, inverse_func=log.log_inverse_transform,
            #                                             check_inverse=True))])
            log_transform = Pipeline([('sqrttransform',
                                    FunctionTransformer(transformations.square_transform))])
            log_transform_pipeline = ColumnTransformer([
                ('_sqrt', Pipeline([('sqrt_transform', log_transform)]), feats)])
            
            temp = pd.DataFrame(data_in.loc[:, feats])
            if not temp.isna().any().any():
                print('Square Root Transform in Progress')
                op = log_transform_pipeline.transformers[0][0]
                key = [[op + '_' + x for x in feats if not str(x) == "nan"]]
                print(key)
                df=use_operation(user_name)
                if not key in df['Key'].to_list():
                    print('New Operation')
                    temp.loc[:, :] = log_transform_pipeline.fit_transform(temp)
                    temp = temp.loc[:, :].add_suffix(op)
                    print(temp)
                    #print(pd.concat([data_in, temp], axis=1))
                    store_temp_operation(log_transform_pipeline, feats)
                    temp = pd.concat([data_in, temp], axis=1)
                else:
                    messages.error(request, 'Operation Already Performed')
                    print('Operation already performed')
                    temp = data_in.copy()
            else:
                print('Not happening')
                messages.error(request, 'Null Values Found in ',feats)
                messages.error(request, feats)
                temp = data_in.copy()
            print(temp)
            # update_train_file(temp_file, data_in)
            columns = [{'field': f, 'title': temp[f].dtypes} for f in temp.columns.to_list()]
            transform_data = {
                'transform_data': temp.head(20).to_html(index=False, index_names=False),
                'columns': columns,
            }
            # data_in.to_csv(temp_file, index=False)
            return render(request, 'transform.html', transform_data)
        
        if 'Box Cox Transform' in operation_chosen:
            # print('Requests: ',request.POST)
            temp_file = current_file(user_name)
            # data_in=pd.read_csv(temp_file)
            data_in = get_current_file(user_name, temp_file)
            
            feats = request.POST.getlist('cols')
            print('Box Cox Transform on ',feats)
            # log_transform = Pipeline([('logtransform',
            #                         FunctionTransformer(log.log_transform, inverse_func=log.log_inverse_transform,
            #                                             check_inverse=True))])
            log_transform = Pipeline([('boxcoxtransform',
                                    FunctionTransformer(transformations.box_cox_transform))])
            log_transform_pipeline = ColumnTransformer([
                ('_boxcox', Pipeline([('boxcox_transform', log_transform)]), feats)])
            
            temp = pd.DataFrame(data_in.loc[:, feats])
            if not temp.isna().any().any():
                print('Box Cox Transform in Progress')
                op = log_transform_pipeline.transformers[0][0]
                key = [[op + '_' + x for x in feats if not str(x) == "nan"]]
                print(key)
                df=use_operation(user_name)
                if not key in df['Key'].to_list():
                    print('New Operation')
                    temp.loc[:, :] = log_transform_pipeline.fit_transform(temp)
                    temp = temp.loc[:, :].add_suffix(op)
                    print(temp)
                    #print(pd.concat([data_in, temp], axis=1))
                    store_temp_operation(log_transform_pipeline, feats)
                    temp = pd.concat([data_in, temp], axis=1)
                else:
                    messages.error(request, 'Operation Already Performed')
                    print('Operation already performed')
                    temp = data_in.copy()
            else:
                print('Not happening')
                messages.error(request, 'Null Values Found in ',feats)
                messages.error(request, feats)
                temp = data_in.copy()
            print(temp)
            # update_train_file(temp_file, data_in)
            columns = [{'field': f, 'title': temp[f].dtypes} for f in temp.columns.to_list()]
            transform_data = {
                'transform_data': temp.head(20).to_html(index=False, index_names=False),
                'columns': columns,
            }
            # data_in.to_csv(temp_file, index=False)
            return render(request, 'transform.html', transform_data)

        if 'Remove Outlier' in operation_chosen:
            # print('Requests: ',request.POST)
            temp_file = current_file(user_name)
            # data_in=pd.read_csv(temp_file)
            data_in = get_current_file(user_name, temp_file)
            feats = request.POST.getlist('cols')
            remove_outliers = Pipeline([('remove_outliers',
                            FunctionTransformer(remove_outlier_IQR))])
            remove_outliers_pipeline = ColumnTransformer([
                ('remove_outliers', Pipeline([('remove_outliers', remove_outliers)]), feats)])

            data_in[feats]=remove_outliers_pipeline.fit_transform(data_in[feats])
            print('Outliers: {0:0.1f} %'.format((data_in[data_in[feats].isna().any(axis=1)].shape[0]/data_in.shape[0])*100))
            messages.success(request, 'Outliers: {0:0.1f} %'.format((data_in[data_in[feats].isna().any(axis=1)].shape[0]/data_in.shape[0])*100))
            data_in=data_in.dropna(subset=feats,axis=0)
            store_temp_operation(remove_outliers_pipeline, feats)
            plt.clf()
            #sns.boxplot(data_in[feats])  # , ax=ax)
            data_in.boxplot(column=feats)
            plt.tight_layout()
            distribution = plot()
            # print(Zscore_outlier(data_in.loc[:,feats]))
            columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
            transform_data = {
                'transform_data': data_in.head(20).to_html(index=False, index_names=False)
                , 'columns': columns,
                'outliers': distribution
            }
            print('Print after view: ', request.POST.getlist('cols'))
            # data_in.to_csv(temp_file, index=False)
            return render(request, 'transform.html', transform_data)

        if 'Median Impute Outliers' in operation_chosen:
            
            temp_file = current_file(user_name)
            # data_in=pd.read_csv(temp_file)
            data_in = get_current_file(user_name, temp_file)
            feats = request.POST.getlist('cols')
            remove_outliers = Pipeline([('remove_outliers',
                            FunctionTransformer(remove_outlier_IQR))])
            remove_outliers_pipeline = ColumnTransformer([
                ('remove_outliers', Pipeline([('median_impute_outliers', remove_outliers)]), feats)])

            data_in[feats]=see_outliers.fit_transform(data_in[feats])
            print('Outliers: {0:0.1f} %'.format((data_in[data_in[feats].isna().any(axis=1)].shape[0]/data_in.shape[0])*100))
            store_temp_operation(remove_outliers, feats)
            plt.clf()
            #sns.boxplot(data_in[feats])  # , ax=ax)
            data_in.boxplot(column=feats)
            plt.tight_layout()
            distribution = plot()
            # print(Zscore_outlier(data_in.loc[:,feats]))
            columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
            transform_data = {
                'transform_data': data_in.head(20).to_html(index=False, index_names=False)
                , 'columns': columns,
                'outliers': distribution
            }
            print('Print after view: ', request.POST.getlist('cols'))
            # data_in.to_csv(temp_file, index=False)
            return render(request, 'transform.html', transform_data)

        if 'Mean Impute Outliers' in operation_chosen:
            
            temp_file = current_file(user_name)
            # data_in=pd.read_csv(temp_file)
            data_in = get_current_file(user_name, temp_file)
            feats = request.POST.getlist('cols')
            see_outliers = Pipeline([('int_transform',
                            FunctionTransformer(remove_outlier_IQR)),
                            ('mean_impute', SimpleImputer(strategy='mean',add_indicator=False))])
            see_outliers_pipeline = ColumnTransformer([
                ('_outliers', Pipeline([('mean_impute_outliers', see_outliers)]), feats)])

            data_in[feats]=see_outliers.fit_transform(data_in[feats])
            print('Outliers: {0:0.1f} %'.format((data_in[data_in[feats].isna().any(axis=1)].shape[0]/data_in.shape[0])*100))
            store_temp_operation(see_outliers_pipeline, feats)
            plt.clf()
            #sns.boxplot(data_in[feats])  # , ax=ax)
            data_in.boxplot(column=feats)
            plt.tight_layout()
            distribution = plot()
            # print(Zscore_outlier(data_in.loc[:,feats]))
            columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
            transform_data = {
                'transform_data': data_in.head(20).to_html(index=False, index_names=False)
                , 'columns': columns,
                'outliers': distribution
            }
            print('Print after view: ', request.POST.getlist('cols'))
            # data_in.to_csv(temp_file, index=False)
            return render(request, 'transform.html', transform_data)


        if 'convert_to_int' in operation_chosen:
            # print('Requests: ',request.POST)
            print('Converting to Integer\n')
            temp_file = current_file(user_name)
            # data_in=pd.read_csv(temp_file)
            data_in = get_current_file(user_name, temp_file)
            feats = request.POST.getlist('cols')

            int_transform = Pipeline([('int_transform',
                                    FunctionTransformer(type_change.to_int))])
            int_transform_pipeline = ColumnTransformer([
                ('_int', Pipeline([('int_transform', int_transform)]), feats)])

            op = int_transform_pipeline.transformers[0][0]
            temp = pd.DataFrame(data_in.loc[:, feats])
            temp.loc[:, :] = int_transform_pipeline.fit_transform(temp.loc[:, :])
            temp = temp.loc[:, :].add_suffix(op)
            temp = pd.concat([data_in, temp], axis=1)
            print('Final Concat: ',temp)
            store_temp_operation(int_transform_pipeline, feats)
            print(temp.loc[:, feats])

            columns = [{'field': f, 'title': temp[f].dtypes} for f in temp.columns.to_list()]
            transform_data = {
                'transform_data': temp.head(20).to_html(index=False, index_names=False)
                , 'columns': columns,
                # 'distribution':distribution
            }
            print('Print after view: ', request.POST.getlist('cols'))
            #update_train_file(temp_file, data_in)
            # data_in.to_csv(temp_file, index=False)
            return render(request, 'transform.html', transform_data)

        if 'convert_to_float' in operation_chosen:
            # print('Requests: ',request.POST)
            print('Converting to Float\n')
            temp_file = current_file(user_name)
            # data_in=pd.read_csv(temp_file)
            data_in = get_current_file(user_name, temp_file)
            feats = request.POST.getlist('cols')

            float_transform = Pipeline([('float_transform',
                                    FunctionTransformer(type_change.to_float))])
            float_transform_pipeline = ColumnTransformer([
                ('_float', Pipeline([('float_transform', float_transform)]), feats)])


            op = float_transform_pipeline.transformers[0][0]
            temp = pd.DataFrame(data_in.loc[:, feats])
            temp.loc[:, :] = float_transform_pipeline.fit_transform(temp.loc[:, :])
            temp = temp.loc[:, :].add_suffix(op)
            temp = pd.concat([data_in, temp], axis=1)
            print('Final Concat: ',temp)
            store_temp_operation(float_transform_pipeline, feats)
            print(temp.loc[:, feats])

            columns = [{'field': f, 'title': temp[f].dtypes} for f in temp.columns.to_list()]
            transform_data = {
                'transform_data': temp.head(20).to_html(index=False, index_names=False)
                , 'columns': columns,
                # 'distribution':distribution
            }
            print('Print after view: ', request.POST.getlist('cols'))
            #update_train_file(temp_file, data_in)
            return render(request, 'transform.html', transform_data)
        
        if 'convert_to_string' in operation_chosen:
            # print('Requests: ',request.POST)
            print('Converting to String\n')
            temp_file = current_file(user_name)
            # data_in=pd.read_csv(temp_file)
            data_in = get_current_file(user_name, temp_file)
            feats = request.POST.getlist('cols')

            int_transform = Pipeline([('string_transform',
                                    FunctionTransformer(type_change.to_string))])
            int_transform_pipeline = ColumnTransformer([
                ('_string', Pipeline([('string_transform', int_transform)]), feats)])

            op = int_transform_pipeline.transformers[0][0]
            temp = pd.DataFrame(data_in.loc[:, feats])
            temp.loc[:, :] = int_transform_pipeline.fit_transform(temp.loc[:, :])
            temp = temp.loc[:, :].add_suffix(op)
            temp = pd.concat([data_in, temp], axis=1)
            print('Final Concat: ',temp)
            store_temp_operation(int_transform_pipeline, feats)
            print(temp.loc[:, feats])

            columns = [{'field': f, 'title': temp[f].dtypes} for f in temp.columns.to_list()]
            transform_data = {
                'transform_data': temp.head(20).to_html(index=False, index_names=False)
                , 'columns': columns,
                # 'distribution':distribution
            }
            print('Print after view: ', request.POST.getlist('cols'))
            update_train_file(temp_file, data_in)
            # data_in.to_csv(temp_file, index=False)
            return render(request, 'transform.html', transform_data)

        if 'Median Impute' in operation_chosen:
            print('Median Imputation\n')
            temp_file = current_file(user_name)
            # data_in = pd.read_csv(temp_file)
            data_in = get_current_file(user_name, temp_file)
            feats = request.POST.getlist('cols')
            print(data_in.loc[:, feats])

            impute_pipeline = ColumnTransformer([
                ('_median_imputed', Pipeline([('median_impute', SimpleImputer(strategy='median', add_indicator=False))]), feats)])
            

            op = impute_pipeline.transformers[0][0]
            key = [[op + '_' + x for x in feats if not str(x) == "nan"]]
            print(key)
            df=use_operation(user_name)
            if not key in df['Key'].to_list():
                temp = pd.DataFrame(data_in.loc[:, feats])
                temp.loc[:, :] = impute_pipeline.fit_transform(temp.loc[:, :])
                temp = temp.loc[:, :].add_suffix(op)
                temp = pd.concat([data_in, temp], axis=1)
                print('Final Concat: ',temp)
                store_temp_operation(impute_pipeline, feats)    
            else:
                messages.error(request, 'Operation Already Performed')
                print('Operation already performed')
                temp = data_in.copy()    


            columns = [{'field': f, 'title': temp[f].dtypes} for f in temp.columns.to_list()]
            transform_data = {
                'transform_data': temp.head(20).to_html(index=False, index_names=False)
                , 'columns': columns,
                # 'distribution':distribution
            }
            print('Print after view: ', request.POST.getlist('cols'))
            #update_train_file(temp_file, data_in)
            # data_in.to_csv(temp_file, index=False)
            return render(request, 'transform.html', transform_data)

        if 'Mean Impute' in operation_chosen:
            print('Mean Imputation\n')
            temp_file = current_file(user_name)
            # data_in = pd.read_csv(temp_file)
            data_in = get_current_file(user_name, temp_file)
            feats = request.POST.getlist('cols')
            impute_pipeline = ColumnTransformer([
                ('_mean_imputed', Pipeline([('mean_impute', SimpleImputer(strategy='mean', add_indicator=False))]), feats)])

            op = impute_pipeline.transformers[0][0]
            key = [[op + '_' + x for x in feats if not str(x) == "nan"]]
            print(key)
            df=use_operation(user_name)
            try:
                if not key in df['Key'].to_list():
                    temp = pd.DataFrame(data_in.loc[:, feats])
                    temp.loc[:, :] = impute_pipeline.fit_transform(temp.loc[:, :])
                    temp = temp.loc[:, :].add_suffix(op)
                    temp = pd.concat([data_in, temp], axis=1)
                    print('Final Concat: ',temp)
                    store_temp_operation(impute_pipeline, feats)    
                else:
                    messages.error(request, 'Operation Already Performed')
                    print('Operation already performed')
                    temp = data_in.copy()
            except:
                print('Not happening')
                messages.error(request, 'Mode Impute Unsuccessful')
            print(data_in.loc[:, feats])
            columns = [{'field': f, 'title': temp[f].dtypes} for f in temp.columns.to_list()]
            transform_data = {
                'transform_data': temp.head(20).to_html(index=False, index_names=False)
                , 'columns': columns,
                # 'distribution':distribution
            }
            print('Print after view: ', request.POST.getlist('cols'))
            #update_train_file(temp_file, data_in)
            return render(request, 'transform.html', transform_data)
        if 'Mode Impute' in operation_chosen:
            print('Mode Imputation\n')
            temp_file = current_file(user_name)
            # data_in = pd.read_csv(temp_file)
            data_in = get_current_file(user_name, temp_file)
            feats = request.POST.getlist('cols')
            print(data_in.loc[:, feats].dtypes)
            #if data_in.loc[:, feats].dtypes.all()=='object':
            impute_pipeline = ColumnTransformer([
                ('_mode_imputed', Pipeline([('mode_impute', SimpleImputer(strategy='most_frequent', add_indicator=False))]),
                feats)])
            try:
                
                op = impute_pipeline.transformers[0][0]
                key = [[op + '_' + x for x in feats if not str(x) == "nan"]]
                print(key)
                df=use_operation(user_name)
                if not key in df['Key'].to_list():
                    temp = pd.DataFrame(data_in.loc[:, feats])
                    temp.loc[:, :] = impute_pipeline.fit_transform(temp.loc[:, :])
                    temp = temp.loc[:, :].add_suffix(op)
                    temp = pd.concat([data_in, temp], axis=1)
                    print('Final Concat: ',temp)
                    store_temp_operation(impute_pipeline, feats)
                else:
                    messages.error(request, 'Operation Already Performed')
                    print('Operation already performed')
                    temp = data_in.copy()
            except:
                print('Not happening')
                messages.error(request, 'Mode Impute Unsuccessful')
            # store_temp_operation(impute_pipeline, feats)
            print(temp.loc[:, feats])

            # else:
            # print('Object Type is not Categorical')
            columns = [{'field': f, 'title': temp[f].dtypes} for f in temp.columns.to_list()]
            transform_data = {
                'transform_data': temp.head(20).to_html(index=False, index_names=False)
                , 'columns': columns,
                # 'distribution':distribution
            }
            
            return render(request, 'transform.html', transform_data)

        if 'Zero Impute' in operation_chosen:
            print('Mean Imputation\n')
            temp_file = current_file(user_name)
            # data_in = pd.read_csv(temp_file)
            data_in = get_current_file(user_name, temp_file)
            feats = request.POST.getlist('cols')
            zero_imp = SimpleImputer(missing_values=np.NaN, strategy='constant', fill_value=0)
            zero_impute_pipeline = ColumnTransformer([
                ('_zero_imputed', Pipeline([('zero_impute', zero_imp)]), feats)])
            data_in.loc[:, feats] = zero_impute_pipeline.fit_transform(data_in.loc[:, feats])
            store_temp_operation(zero_impute_pipeline, feats)
            print(data_in.loc[:, feats])
            columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
            transform_data = {
                'transform_data': data_in.head(20).to_html(index=False, index_names=False)
                , 'columns': columns,
                # 'distribution':distribution
            }
            print('Print after view: ', request.POST.getlist('cols'))
            # data_in.to_csv(temp_file, index=False)
            update_train_file(temp_file, data_in)
            return render(request, 'transform.html', transform_data)
    
    if 'symbol_removal' in request.POST and request.method == "POST":
        # print('Requests: ',request.POST)
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        feats = request.POST.getlist('cols')
        print('Symbol Removal in Progress on : ', feats)

        symbol_removal = Pipeline([('symbol', FunctionTransformer(symbol_remove))])
        symbol_removal_pipeline = ColumnTransformer([
            ('_symbol_removed', Pipeline([('symbol_removal', symbol_removal)]), feats)])
        print(symbol_removal)
        data_in.loc[:, feats] = symbol_removal_pipeline.fit_transform(data_in.loc[:, feats])
        store_temp_operation(symbol_removal_pipeline, feats)
        update_train_file(temp_file, data_in)
        columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        transform_data = {
            'transform_data': data_in.head(20).to_html(index=False, index_names=False)
            , 'columns': columns
        }
        # data_in.to_csv(temp_file, index=False)
        return render(request, 'transform.html', transform_data)

    if 'date_time' in request.POST and request.method == "POST":
        # print('Requests: ',request.POST)
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        feats = request.POST.getlist('cols')
        print('Date Time Conversion in Progress on : ', feats)
        print(data_in.loc[:, feats].head())
        date_format = Pipeline([('date_format', FunctionTransformer(datetimeformat))])
        date_format_pipeline = ColumnTransformer([
            ('_date_formated', Pipeline([('date_format', date_format)]), feats)])
        data_in.loc[:, feats] = date_format_pipeline.fit_transform(data_in.loc[:, feats])
        columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        transform_data = {
            'transform_data': data_in.head(20).to_html(index=False, index_names=False)
            , 'columns': columns
        }
        # data_in.to_csv(temp_file, index=False)
        update_train_file(temp_file, data_in)
        return render(request, 'transform.html', transform_data)

    if 'drop_column' in request.POST and request.method == "POST":
        # print('Requests: ',request.POST)
        temp_file = current_file(user_name)
        project_id=get_current_project(user_name)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        feats = request.POST.getlist('cols')
        print('Removing : ', feats)
        #data_in = data_in.drop(feats, axis=1)
        # print(data_in.loc[:,feats].head())
        temporary_data=data_in.loc[:,feats]
        temporary(user_name, temporary_data)
        data_in=drop_columns(data_in,feats)
        store_temp_operation(drop_columns, feats)
        operations = use_operation(user_name)
        temp_operations = retrieve_temp_operation(user_name).reset_index(drop=True)
        print('vdvddvdvdvd',temp_operations)
        operations = pd.concat([operations, temp_operations], ignore_index=True)
        print('Concat Operation :',operations)
        store_all_operation(user_name, temp_file, operations)
        target = get_target(user_name, temp_file,project_id)
        selected_columns = pd.DataFrame({'Selected by SelectKBest': data_in.columns})
        print(selected_columns)

        save_columns_selected(user_name, temp_file, selected_columns.to_json())
        update_train_file(user_name, temp_file, data_in)
        columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        transform_data = {
            'transform_data': data_in.head(20).to_html(index=False, index_names=False)
            , 'columns': columns
        }
        # data_in.to_csv(temp_file, index=False)
        return render(request, 'transform.html', transform_data)
    if 'multicoll' in request.POST and request.method == "POST":
        temp_file=current_file(user_name)
        data_in = get_current_file(user_name, temp_file)
        plt.clf()
        feats = data_in.select_dtypes(exclude=[object]).columns.values
        if len(feats) == 0:
            feats = data_in.columns
        print(feats)
        vif_data = pd.DataFrame()
        vif_data["Feature"] = feats
        # calculating VIF for each feature
        vif_data["VIF"] = [variance_inflation_factor(data_in.values, i)
                        for i in range(len(data_in[feats].columns))]
        print(vif_data)
        vif_data=vif_data.sort_values(by="VIF",ascending=False)
        plt.figure(figsize=(10, 10))
        plots = sns.barplot(y=vif_data["Feature"], x=vif_data["VIF"])#, order=vif_data.sort_values(by="VIF")["Feature"])
        for p in plots.patches:
            plots.annotate("%.4f" % p.get_width(), xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                        xytext=(5, 0), textcoords='offset points', ha="left", va="center")
        plt.tight_layout()
        distribution = plot()
        print(vif_data)
        columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        transform_data = {
            'transform_data': data_in.head(20).to_html(index=False, index_names=False),
            'columns': columns,
            'distribution':distribution
        }
        return render(request, 'transform.html', transform_data)

    if 'feat_importance' in request.POST and request.method == "POST":
        # print('Requests: ',request.POST)
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        target = get_target(user_name, temp_file,project_id)
        print(target)
        transform_data = pd.DataFrame({'Correlation': data_in.corr()[target]})
        transform_data = transform_data.rename_axis('Columns').reset_index(inplace=False)

        columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        transform_data = {
            'transform_data': transform_data.to_html(index=False, index_names=False)
            , 'columns': columns,
        }
        print('Print after view: ', request.POST.getlist('cols'))
        # data_in.to_csv(temp_file, index=False)
        return render(request, 'transform.html', transform_data)
    if 'view_steps' in request.POST and request.method == "POST":
        temp_file = current_file(user_name)
        # transform=pd.read_csv(temp_file)
        transform = get_current_file(user_name, temp_file)
        columns = [{'field': f, 'title': transform[f].dtypes} for f in transform.columns.to_list()]

        feat_transforms = use_operation(user_name)
        print(feat_transforms)
        # print([f.transformers[0][1][0] if type(f).__name__=='ColumnTransformer' else f.__name__ for f in feat_transforms['Operation']])
        # feat_transforms['Operations']=[f.transformers[0][1][0].steps[0][0] if type(f).__name__=='ColumnTransformer' else f.__name__ type(list(f.values())[0]).__name__ if type(f).__name__=='defaultdict' else f.__name__ for f in feat_transforms['Operation']]
        operations = []
        # if type(feat_transforms)!='NoneType':
        for f in feat_transforms['Operation']:
            if type(f).__name__ == 'ColumnTransformer':
                try:
                    operations.append(f.transformers[0][1][0].steps[0][0])
                except:
                    operations.append(type(f.transformers[0][1][0]).__name__)
            elif type(f).__name__ == 'defaultdict':
                operations.append(type(list(f.values())[0]).__name__)
            else:
                operations.append(f.__name__)
        feat_transforms['Operations'] = operations
        print(feat_transforms.loc[:, ['Operations', 'Column']])
        feat_transforms = [{'field': f, 'title': d} for f, d in
                        zip(feat_transforms['Operations'], feat_transforms['Column'])]
        print(feat_transforms)

        transform_data = {
            # 'feat_transforms':feat_transforms.loc[:,['Operations','Column']].to_html(index=False, index_names=False),
            'columns': columns,
            'feat_transforms': feat_transforms
            # 'transform_data':transform.head(20).to_html(index=False, index_names=False)
        }

        return render(request, "transform.html", transform_data)

    if 'submit_steps' in request.POST and request.method == "POST":
        temp_file = current_file(user_name)
        # transform=pd.read_csv(temp_file)
        transform = get_current_file(user_name, temp_file)
        

        operations = use_operation(user_name)
        print(operations)
        temp_operations = retrieve_temp_operation(user_name).reset_index(drop=True)
        column_list =temp_operations['Column'].tolist()
        feats = list(set(reduce(lambda x,y: x+y, column_list)))
        
        temporary_data=transform.loc[:,feats]
        temporary(user_name, temporary_data)
        
        #store_operation(temp_operations['Operation'][0],temp_operations['Column'][0])
        operations = pd.concat([operations, temp_operations], ignore_index=True)
        print('Concatenated: ')
        print(operations)
        store_all_operation(user_name, temp_file, operations)
        transform = apply_operations(transform, temp_operations)
        columns = [{'field': f, 'title': transform[f].dtypes} for f in transform.columns.to_list()]
        update_train_file(user_name, temp_file, transform)
        # transform.to_csv(temp_file, index=False)
        transform_data = {
            'columns': columns,
            'transform_data': transform.head(20).to_html(index=False, index_names=False)
        }

        return render(request, "transform.html", transform_data)
    if 'remove_steps' in request.POST and request.method == "POST":
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        # columns=[{'field':f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        print('Chosen Option: ', request.POST.getlist('trans'))
        index = request.POST.getlist('trans')
        index = [int(i) for i in index]
        print('Chosen Option: ', index)
        feat_engine = use_operation(user_name)

        print('Chosen: ', feat_engine.loc[index, :])
        remove_steps = feat_engine.loc[index, :]
        feat_engine = feat_engine.drop(feat_engine.loc[index, :].index)
        print(feat_engine)
        store_all_operation(user_name, temp_file, feat_engine.reset_index(drop=True))
        #feat_engine.to_pickle(os.path.join(temp_dir, 'operations.pkl'))
        # apply_operations(test_data)
        data_in = apply_reverse_operations(user_name, data_in, remove_steps)
        print('Transformed_Data', data_in)

        columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        update_train_file(user_name, temp_file, data_in)
        temp_data_remove(user_name, remove_steps['Column'].reset_index(drop=True)[0])
        
        
        transform_data = {
            'columns': columns,
            'transform_data': data_in.head(20).to_html(index=False, index_names=False)
        }

        return render(request, "transform.html", transform_data)

    if 'view_transform_data' in request.POST and request.method == "POST":
        print('Transform Data')
        feat_engine = use_operation(user_name)
        some_list =feat_engine['Column'].tolist()
        single_list = list(set(reduce(lambda x,y: x+y, some_list)))
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        data_final=data_in[single_list]
        columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        print(data_final)
        transform_data = {
            'columns': columns,
            'transform_data': data_final.head(20).to_html(index=False, index_names=False)
        }
        return render(request, "transform.html", transform_data)

    else:
        temp_file = current_file(user_name)
        # transform=pd.read_csv(temp_file)
        project_id=get_current_project(user_name)
        query=f"select project_name from public.base_data where user_name='{user_name}' AND project_id='{project_id}'".format(user_name,project_id)
        project_name=pd.read_sql_query(query, sql_engine()).reset_index(drop="True")['project_name'][0]
        # project_name=sql_engine()(query, return_data=True).reset_index(drop="True")['project_name'][0]
        #transform = get_current_file(user_name, temp_file)
        transform = get_current_display_file(user_name, temp_file)
        columns = [{'field': f, 'title': transform[f].dtypes} for f in transform.columns.to_list()]
        transform_data = {
            'columns': columns,
            # 'transform_data': transform.head(20).to_html(index=False, index_names=False),
            'transform_data': transform.head(20).to_dict(orient='records'),
            'columns':transform.columns.values,
            'project_name':project_name
        }

        return render(request, "transform.html", transform_data)


def visual(request):        
    user_name = request.session['user_id']
    if 'distplot' in request.POST and request.method == "POST":
        # print('Requests: ',request.POST)
        print(current_file(user_name))
        temp_file = current_file(user_name)
        data_in = get_current_file(user_name, temp_file)
        project_id = get_current_project(user_name)
        #file_name = psql.read_sql_query("select file_name from public.train_data where use_file='Yes'", engine).reset_index(drop=True)['file_name'][0]
        # data_in=pd.read_csv(temp_file)
        # feats=request.POST.getlist('cols')
        col1 = request.POST['xaxis']
        col2 = request.POST['yaxis']
        colorby = request.POST['colorby']
        store_chart(user_name,project_id, temp_file, 'distplot', x=col1)
        # print(feats)
        l = (len(data_in.columns) // 3) + (len(data_in.columns) % 3)
        plt.clf()
        plt.title('Histogram of ' + col1 + ' column')

        fig = px.histogram(data_in, x=data_in[col1], marginal="box", hover_data=data_in.columns)
        distribution = fig.to_html(full_html=False, default_height=500, default_width=700)
        # distribution=plot()
        feats = []
        columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        
        distribution = {'distribution': distribution, 'columns': columns, 'columns1': columns, 'columns2': columns}
        # return render(request, 'transform.html', distribution)
        return render(request, "visual.html", distribution)

    if 'countplot' in request.POST and request.method == "POST":
        # print('Requests: ',request.POST)
        temp_file = current_file(user_name)
        data_in = get_current_file(user_name, temp_file)
        # data_in=pd.read_csv(temp_file)
        plt.clf()
        # l=(len(data_in.columns)//3)+(len(data_in.columns)%3)

        col1 = request.POST['xaxis']
        col2 = request.POST['yaxis']

        # colorby=None
        colorby = request.POST['colorby']
        print(colorby)
        colorby = None if colorby == '' else colorby

        plt.figure(figsize=(12, 5))
        # sns.countplot(x=col1, data=data_in, palette=sns.color_palette("cubehelix"))
        fig = px.histogram(data_in, x=col1, color=colorby, barmode='group')

        distribution = fig.to_html(full_html=False, default_height=500, default_width=700)
        columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        # columns=[{'field':f, 'title': f} for f in data_in.columns.to_list()]
        distribution = {'distribution': distribution, 'columns': columns, 'columns1': columns, 'columns2': columns, }
        return render(request, 'visual.html', distribution)

    if 'scatterplot' in request.POST and request.method == "POST":
        # print('Requests: ',request.POST)
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        project_id = get_current_project(user_name)
        #file_name = psql.read_sql_query("select file_name from public.train_data where use_file='Yes'", engine).reset_index(drop=True)['file_name'][0]
        plt.clf()
        # l=(len(data_in.columns)//3)+(len(data_in.columns)%3)
        #feats = request.POST.getlist('cols')
        col1 = request.POST['xaxis']
        col2 = request.POST['yaxis']
        colorby = request.POST['colorby']
        store_chart(user_name,project_id, temp_file, 'scatterplot', x=col1, y=col2)
        # fig, axes = plt.subplots( figsize=(l, l*3),nrows=l, ncols=3,squeeze=True)
        plt.figure(figsize=(7, 7))
        #plt.title('Scatterplot of ' + str.join(' & ', feats) + ' column')
        fig = px.scatter(data_in, x=data_in[col1], y=data_in[col2], color = colorby)
        distribution = fig.to_html(full_html=False, default_height=500, default_width=700)
        # distribution=plot()
        columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        # columns=[{'field':f, 'title': f} for f in data_in.columns.to_list()]
        distribution = {'distribution': distribution, 'columns': columns, 'columns1': columns, 'columns2': columns}
        return render(request, 'visual.html', distribution)

    if 'lineplot' in request.POST and request.method == "POST":
        # print('Requests: ',request.POST)
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        plt.clf()
        # l=(len(data_in.columns)//3)+(len(data_in.columns)%3)
        # feats=request.POST.getlist('cols')
        col1 = request.POST['xaxis']
        col2 = request.POST['yaxis']
        # fig, axes = plt.subplots( figsize=(l, l*3),nrows=l, ncols=3,squeeze=True)
        plt.figure(figsize=(7, 7))
        fig = px.line(data_in, x=data_in[col1], y=data_in[col2])
        distribution = fig.to_html(full_html=False, default_height=500, default_width=700)
        # distribution=plot()
        columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        distribution = {'distribution': distribution, 'columns': columns, 'columns1': columns, 'columns2': columns}
        return render(request, 'visual.html', distribution)

    if 'boxplot' in request.POST and request.method == "POST":
        print('Box Plot')
        # print('Requests: ',request.POST)
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        plt.clf()
        # l=(len(data_in.columns)//3)+(len(data_in.columns)%3)
        # feats=request.POST.getlist('cols')
        col1 = request.POST['xaxis']
        col2 = request.POST['yaxis']
        # fig, axes = plt.subplots( figsize=(l, l*3),nrows=l, ncols=3,squeeze=True)
        plt.figure(figsize=(7, 7))
        # plt.title('Scatterplot of '+str.join(' & ',feats)+' column')
        fig = px.box(data_in, y=data_in[col1], x=data_in[col2])
        distribution = fig.to_html(full_html=False, default_height=500, default_width=700)
        # distribution=plot()
        columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        # columns=[{'field':f, 'title': f} for f in data_in.columns.to_list()]
        distribution = {'distribution': distribution, 'columns': columns, 'columns1': columns, 'columns2': columns}
        return render(request, 'visual.html', distribution)

    if 'piechart' in request.POST and request.method == "POST":
        # print('Requests: ',request.POST)
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        plt.clf()
        # l=(len(data_in.columns)//3)+(len(data_in.columns)%3)
        col1 = request.POST['xaxis']
        col2 = request.POST['colorby']
        print(col1, col2)
        # fig, axes = plt.subplots( figsize=(l, l*3),nrows=l, ncols=3,squeeze=True)
        plt.figure(figsize=(7, 7))
        # plt.title('Scatterplot of '+str.join(' & ',feats)+' column')
        fig = px.pie(data_in, values=data_in[col1], names=data_in[col2])
        distribution = fig.to_html(full_html=False, default_height=500, default_width=700)
        # distribution=plot()
        columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        # columns=[{'field':f, 'title': f} for f in data_in.columns.to_list()]
        distribution = {'distribution': distribution, 'columns': columns, 'columns1': columns, 'columns2': columns}
        return render(request, 'visual.html', distribution)

    if 'correlation' in request.POST and request.method == "POST":
        # print('Requests: ',request.POST)
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        # feats=request.POST.getlist('cols')
        col1 = request.POST['xaxis']
        print(col1)
        plt.figure(figsize=(14, 8))
        if (len(col1) == 0):
            # sns.heatmap(data_in.corr(), annot=True, fmt='.2g', mask=np.triu(data_in.corr()), vmax=1, vmin=-1, center=0)
            fig = px.imshow(data_in.corr(), color_continuous_scale=px.colors.diverging.balance,
                            labels=dict(color='Correlation'))
            fig.update_layout(autosize=False, width=800, height=800)
        else:
            # sns.heatmap(data_in.corr()[feats], annot=True, fmt='.2g', vmax=1, vmin=-1, center=0)
            #fig = px.imshow(pd.DataFrame(data_in.corr()[col1]), color_continuous_scale=px.colors.diverging.balance)
            corrs=pd.DataFrame(data_in.corr()[col1])
            corrs=corrs.loc[set(corrs.index) - set(corrs.loc[col1,:].index),:].sort_values(by=col1, ascending=True, key=abs)
            fig = px.bar(data_frame=corrs, x=col1,orientation='h',color_continuous_scale='Bluered_r', color=col1, range_color=[-1,1])
            fig.update_layout(autosize=False, width=1000, height=500)
        distribution = fig.to_html(full_html=False, default_height=500, default_width=700)
        # plt.tight_layout()
        # distribution=plot()
        columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        distribution = {'distribution': distribution, 'columns': columns, 'columns1': columns, 'columns2': columns}
        return render(request, 'visual.html', distribution)
    if 'coldescrib' in request.POST and request.method == "POST":
        # print('Requests: ',request.POST)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        feats = request.POST.getlist('cols')
        print(feats)
        plt.figure(figsize=(14, 20))
        fig, axs = plt.subplots(nrows=4, figsize=(10, 15))
        sns.distplot(data_in[feats[0]], ax=axs[0])
        sns.boxplot(data_in[feats[0]], ax=axs[1])
        if data_in[feats[0]].dtype == 'object':
            sns.countplot(data_in[feats[0]], ax=axs[2])
        else:
            fig.delaxes(axs[2])
        sns.heatmap(data_in.corr()[feats], annot=True, fmt='.2g', vmax=1, vmin=-1, center=0)
        plt.tight_layout()
        distribution = plot()
        des = data_in.describe(include='all')[feats[0]]
        des['Data Type'] = data_in[feats[0]].dtypes
        des['Missing Observations'] = data_in[feats[0]].isna().sum()
        des = pd.DataFrame(des).T
        data_described = des.to_html(index=False)
        columns = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        distribution = {'data_described': data_described, 'distribution': distribution, 'columns': columns}
        return render(request, 'visual.html', distribution)

    else:
        temp_file = current_file(user_name)
        transform = get_current_display_file(user_name, temp_file)
        # transform=pd.read_csv(temp_file)
        columns = [{'field': f, 'title': transform[f].dtypes} for f in transform.columns.to_list()]
        transform_data = {
            'columns1': columns, 'columns2': columns,
            'transform_data': transform.to_html(index=False, index_names=False)
        }
        return render(request, "visual.html", transform_data)


def modelling(request):
    user_name = request.session['user_id']
    if 'selectkbest' in request.POST and request.method == "POST":
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        project_id = get_current_project(user_name)
        target = get_target(user_name, temp_file,project_id)
        X_train = data_in.drop(target, axis=1)
        y_train = data_in[target].astype(str)
        no_feat_select = request.POST.get('n_feats')
        print('Number of features :', no_feat_select)
        no_features = data_in.shape[1] - 1
        n_feats = int(request.POST.get('n_feats')) if no_feat_select != '' else no_features
        print('Number of features :', n_feats)
        if no_feat_select != '':
            problem_type = get_problem_type(user_name, temp_file)
            print('Problem Type: ', problem_type)
            mutual_info = mutual_info_classif if problem_type == 'classification' else mutual_info_regression
            print('Starting SelectKBest')
            sel_best_cols = SelectKBest(mutual_info, k=n_feats)
            print('Select K Best in Progress')
            sel_best_cols.fit(X_train, y_train)
            print(X_train.columns[sel_best_cols.get_support()])
            selected_columns = pd.DataFrame({'Selected by SelectKBest': X_train.columns[sel_best_cols.get_support()]})
        else:
            selected_columns = pd.DataFrame({'Selected by SelectKBest': X_train.columns.to_list()})
        #with open(os.path.join(temp_dir, 'columns_selected.obj'), 'wb') as fp:pickle.dump(list(X_train.columns[sel_best_cols.get_support()]), fp)
        save_columns_selected(user_name, temp_file, selected_columns.to_json())
        #feats = request.POST.getlist('cols')
        #print(data_in.loc[:, feats].head())
        columns = [{'field': f, 'title': X_train[f].dtypes} for f in X_train.columns.to_list()]
        target_selected = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]

        print(target)
        #transform_data = pd.DataFrame({'Correlation': data_in.corr()[target]})
        #transform_data = transform_data.rename_axis('Columns').reset_index(inplace=False)
        no_features = data_in.shape[1] - 1
        transform_data = {
            #'transform_data': transform_data.to_html(index=False, index_names=False),
             'selected_columns': selected_columns.to_html(index=False, index_names=False)
            , 'columns': columns
            , 'target_selected': target_selected
            , 'target': target
            , 'no_features': no_features
        }
        return render(request, "modelling.html", transform_data)
    if 'autofeatselect' in request.POST and request.method == "POST":
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        project_id = get_current_project(user_name)
        target = get_target(user_name, temp_file,project_id)
        X_train = data_in.drop(target, axis=1)
        y_train = data_in[target]
        
        problem_type = get_problem_type(user_name, temp_file)
        print('Problem Type: ', problem_type)
        fsel = autofeat.FeatureSelector(problem_type=problem_type)
        print('Starting Auto Feature Selector')
        new_X = fsel.fit_transform(X_train, y_train)
        selected_columns = pd.DataFrame({'Selected by Auto Feature Selector': new_X.columns.to_list()})
        save_columns_selected(user_name, temp_file, selected_columns.to_json())
        # feats=request.POST.getlist('cols')
        # print(data_in.loc[:,feats].head())
        columns = [{'field': f, 'title': X_train[f].dtypes} for f in X_train.columns.to_list()]
        target_selected = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]

        print(target)
        transform_data = pd.DataFrame({'Correlation': data_in.corr()[target]})
        transform_data = transform_data.rename_axis('Columns').reset_index(inplace=False)
        no_features = data_in.shape[1] - 1
        transform_data = {
            'transform_data': transform_data.to_html(index=False, index_names=False)
            , 'selected_columns': selected_columns.to_html(index=False, index_names=False)
            , 'columns': columns
            , 'target_selected': target_selected
            , 'target': target
            , 'no_features': no_features
        }
        return render(request, "modelling.html", transform_data)

    if 'rfecv' in request.POST and request.method == "POST":
        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        project_id = get_current_project(user_name)
        target = get_target(user_name, temp_file,project_id)
        X_train = data_in.drop(target, axis=1)
        y_train = data_in[target]
        with open('problem_type.obj', 'rb') as fp:
            problem_type = pickle.load(fp)
        problem_type = list(problem_type.values())[0]
        problem_type = get_problem_type(user_name, temp_file)
        print('Problem Type: ', problem_type)
        estimator = RandomForestRegressor() if problem_type == 'regression' else RandomForestClassifier()
        fsel = RFECV(estimator, step=1, cv=5)
        fsel = fsel.fit(X_train, y_train)
        columns = X_train[X_train.columns[fsel.get_support(1)]].columns.to_list()
        print(columns)
        selected_columns = pd.DataFrame({'Selected by Recursive Feature Elimination': columns})
        save_columns_selected(user_name, temp_file, selected_columns.to_json())
        # feats=request.POST.getlist('cols')
        columns = [{'field': f, 'title': X_train[f].dtypes} for f in X_train.columns.to_list()]
        target_selected = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]

        print(target)
        transform_data = pd.DataFrame({'Correlation': data_in.corr()[target]})
        transform_data = transform_data.rename_axis('Columns').reset_index(inplace=False)
        no_features = data_in.shape[1] - 1
        transform_data = {
            'transform_data': transform_data.to_html(index=False, index_names=False)
            , 'selected_columns': selected_columns.to_html(index=False, index_names=False)
            , 'columns': columns
            , 'target_selected': target_selected
            , 'target': target
            , 'no_features': no_features
        }
        return render(request, "modelling.html", transform_data)

    if 'save' in request.POST and request.method == "POST":

        print('Selected Columns: ', request.POST.getlist('columns'))

        temp_file = current_file(user_name)
        print(temp_file)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        feats = request.POST.getlist('cols')
        project_id=get_current_project(user_name)
        target = get_target(user_name, temp_file,project_id)
        print(target)
        X_train = data_in.drop(target, axis=1)
        y_train = data_in[target]
        columns = [{'field': f, 'title': X_train[f].dtypes} for f in X_train.columns.to_list()]
        target_selected = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        columns_sel = request.POST.getlist('columns')
        selected_columns = pd.DataFrame({'Selected Columns': columns_sel})
        save_columns_selected(user_name, temp_file, selected_columns.to_json())
        print(data_in.columns)
        print(data_in.corr())
        transform_data = pd.DataFrame({'Correlation': data_in.corr()[target]})
        transform_data = transform_data.rename_axis('Columns').reset_index(inplace=False)
        no_features = data_in.shape[1] - 1
        transform_data = {
            'transform_data': transform_data.to_html(index=False, index_names=False)
            , 'selected_columns': selected_columns.to_html(index=False, index_names=False)
            , 'columns': columns
            , 'target_selected': target_selected
            , 'target': target
            , 'no_features': no_features
        }
        return render(request, "modelling.html", transform_data)

    else:

        temp_file = current_file(user_name)
        print(temp_file)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_display_file(user_name, temp_file)
        feats = request.POST.getlist('cols')

        # target_selected=columns.copy()
        project_id = get_current_project(user_name)
        target = get_target(user_name, temp_file,project_id)
        print(target)
        X_train = data_in.drop(target, axis=1)
        y_train = data_in[target]
        columns = [{'field': f, 'title': X_train[f].dtypes} for f in X_train.columns.to_list()]
        target_selected = [{'field': f, 'title': data_in[f].dtypes} for f in data_in.columns.to_list()]
        r'''
        transform_data = pd.DataFrame({'Correlation': data_in.corr()[target]})
        transform_data = transform_data.rename_axis('Columns').reset_index(inplace=False)
        '''
        no_features = data_in.shape[1] - 1
        transform_data = {
            # 'transform_data': transform_data.to_html(index=False, index_names=False)
            'columns': columns
            , 'target_selected': target_selected
            , 'target': target
            , 'no_features': no_features
        }
        return render(request, "modelling.html", transform_data)
    

class_dict = {'log_reg': LogisticRegression(),
              'rf': RandomForestClassifier(),
              'dt': DecisionTreeClassifier(),
              'bayes': GaussianNB(),
              'knn': KNeighborsClassifier(),
              'adaboost': AdaBoostClassifier(),
              'gb': GradientBoostingClassifier(),
              'xgb': XGBClassifier()}
reg_dict = {'log_reg': LinearRegression(),
            'rf': RandomForestRegressor(),
            'dt': DecisionTreeRegressor(),
            # 'bayes':GaussianNB(),
            'knn': KNeighborsRegressor(),
            'adaboost': AdaBoostRegressor(),
            'gb': GradientBoostingRegressor(),
            'xgb': XGBRegressor()}


def ml_model(request):
    class_validation = {'train_test':'Train-Test Split', 'kfold': 'KFold','stratifiedk_fold':'Stratified KFold'}
    reg_validation = {'train_test':'Train-Test Split','kfold': 'KFold','shuffle_split': 'ShuffleSplit'}
    
    user_name = request.session['user_id']
    if 'submit_model' in request.POST and request.method == "POST":

        temp_file = current_file(user_name)
        # data_in=pd.read_csv(temp_file)
        data_in = get_current_file(user_name, temp_file)
        project_id = get_current_project(user_name)
        target = get_target(user_name, temp_file,project_id)
        print('Target: ', target)
        
        columns_selected = get_columns_selected(user_name, temp_file)
        print('Selected Columns: ', columns_selected)
        project_id = get_current_project(user_name)
        algos = request.POST.getlist('model_select')
        
        tuned_models=get_models(user_name, temp_file)
        if tuned_models.shape[0]>0:
            tuned_models = tuned_models[tuned_models['model_name'].str.startswith('Tuned')]
            print('tuned1 :',tuned_models)
        print(algos)
        folds = int(request.POST['folds'])
        test_size = int(request.POST['test_split'])
        x = data_in.drop(target, axis=1).loc[:, columns_selected]
        print(x.head())
        y = data_in.loc[:, target]
        print(y.head())
        
        print('KFolds : ', folds)
        
        problem_type = get_problem_type(user_name, temp_file)
        print('Problem Type: ', problem_type)
        models, accuracy, precision, recall, f1, roc_auc = [], [], [], [], [], []
        #save_models, params = [], []
        model_files, time_fit=[],[]
        neg_mean_absolute_error, neg_mean_squared_error, neg_root_mean_squared_error, neg_mean_squared_log_error, r2, explained_variance = [], [], [], [], [], []
        ml_dict = reg_dict if problem_type == 'regression' else class_dict
        for i in algos:
            print(i)
            
            if 'train_test' in request.POST['cross_validation']:

                if not i.startswith('Tuned'):
                    print('Algorithm:', type(ml_dict.get(i)).__name__)
                    summary_score = train_test_fit(ml_dict.get(i), x, y, test_size, problem_type,
                                                   request.POST['sampling'])
                    models.append(type(ml_dict.get(i)).__name__)
                    
                #elif i.endswith('.sav')
                elif i.startswith('Tuned'):
                    print('Tuned Model')
                    tuned_model=tuned_models[tuned_models['model_name']==i]['model_file'].reset_index(drop=True)[0]
                    print('Algorithm:', type(tuned_model).__name__)
                    summary_score = train_test_fit(tuned_model, x, y, test_size, problem_type, request.POST['sampling'])
                    models.append(i)
                    
                print(summary_score)
                if problem_type == 'classification':
                    accuracy.append(round(summary_score['accuracy'].mean(), 4))
                    precision.append(round(summary_score['precision_weighted'].mean(), 4))
                    recall.append(round(summary_score['recall_weighted'].mean(), 4))
                    f1.append(round(summary_score['f1_weighted'].mean(), 4))
                    roc_auc.append(round(summary_score['roc_auc_ovr_weighted'].mean(), 4))
                    time_fit.append(str(datetime.datetime.now().strftime('%d/%m/%Y %H-%M-%S %p')))
                    result = pd.DataFrame({'Model': models, 'Accuracy': accuracy,
                                           'Precision ': precision,
                                           'Recall ': recall,
                                           'F1 Score ': f1,
                                           'ROC Score ': roc_auc,
                                           'Fitted Time':time_fit})
                    print(result)
                if problem_type == 'regression':
                    neg_mean_absolute_error.append(round(summary_score['neg_mean_absolute_error'].mean(), 4))
                    neg_mean_squared_error.append(round(summary_score['neg_mean_squared_error'].mean(), 4))
                    neg_root_mean_squared_error.append(round(summary_score['neg_root_mean_squared_error'].mean(), 4))
                    r2.append(round(summary_score['r2'].mean(), 4))
                    explained_variance.append(round(summary_score['explained_variance'].mean(), 4))
                    time_fit.append(str(datetime.datetime.now().strftime('%d/%m/%Y %H-%M-%S %p')))
                    result = pd.DataFrame({'Model': models, 'mean_absolute_error': neg_mean_absolute_error,
                                           'mean_squared_error ': neg_mean_squared_error,
                                           'root_mean_squared_error ': neg_root_mean_squared_error,
                                           'R2 Score ': r2,
                                           'Explained Variance ': explained_variance,
                                           'Fitted Time':time_fit})
            if request.POST['cross_validation'] not in ['train_test']:
                if not i.startswith('Tuned'):
                    # For inbuilt models
                    print('Algorithm:', type(ml_dict.get(i)).__name__)
                    models.append(type(ml_dict.get(i)).__name__)
                    summary_score = pd.DataFrame(fit_ml_algo_predict(ml_dict.get(i), x, y, folds, problem_type,
                                                                     request.POST['cross_validation']))
                elif i.startswith('Tuned'):
                    # For tuned models
                    print('Tuned Model')
                    tuned_model=tuned_models[tuned_models['model_name']==i]['model_file'].reset_index(drop=True)[0]
                    print('Algorithm:', type(tuned_model).__name__)
                    summary_score = pd.DataFrame(fit_ml_algo_predict(tuned_model,
                                                                     x, y, folds, problem_type,
                                                                     request.POST['cross_validation']))
                    models.append(i)

                print(summary_score)
                if problem_type == 'classification':
                    accuracy.append(round(summary_score['test_accuracy'].mean(), 4))
                    precision.append(round(summary_score['test_precision_weighted'].mean(), 4))
                    recall.append(round(summary_score['test_recall_weighted'].mean(), 4))
                    f1.append(round(summary_score['test_f1_weighted'].mean(), 4))
                    roc_auc.append(round(summary_score['test_roc_auc_ovr_weighted'].mean(), 4))
                    time_fit.append(str(datetime.datetime.now().strftime('%d/%m/%Y %H-%M-%S %p')))
                    result = pd.DataFrame({'Model': models, 'Accuracy': accuracy,
                                           'Precision ': precision,
                                           'Recall ': recall,
                                           'F1 Score ': f1,
                                           'ROC Score ': roc_auc,
                                           'Fitted Time':time_fit})
                if problem_type == 'regression':
                    neg_mean_absolute_error.append(np.abs(round(summary_score['test_neg_mean_absolute_error'].mean(), 4)))
                    neg_mean_squared_error.append(np.abs(round(summary_score['test_neg_mean_squared_error'].mean(), 4)))
                    neg_root_mean_squared_error.append(
                        np.abs(round(summary_score['test_neg_root_mean_squared_error'].mean(), 4)))
                    r2.append(round(summary_score['test_r2'].mean(), 4))
                    explained_variance.append(np.abs(round(summary_score['test_explained_variance'].mean(), 4)))
                    time_fit.append(str(datetime.datetime.now().strftime('%d/%m/%Y %H-%M-%S %p')))
                    result = pd.DataFrame({'Model': models, 'mean_absolute_error': neg_mean_absolute_error,
                                           'mean_squared_error ': neg_mean_squared_error,
                                           'root_mean_squared_error ': neg_root_mean_squared_error,
                                           'R2 Score ': r2,
                                           'Explained Variance ': explained_variance,
                                           'Fitted Time':time_fit})

            #filename = os.path.join(temp_dir, i + '.sav')
            # pickle.dump(ml_dict.get(i), open(filename, 'wb'))
            if not i.startswith('Tuned'):
                if request.POST['cross_validation'] not in ['train_test']:
                    ml_dict.get(i).fit(x, y)
                #filename = os.path.join(temp_dir, i + '.sav')
                #joblib.dump(ml_dict.get(i), filename)
                #Add list append of models here
                model_files.append(ml_dict.get(i))
            elif i.startswith('Tuned'):
                print(i)
                if request.POST['cross_validation'] not in ['train_test']:
                    tuned_model.fit(x, y)
                #joblib.dump(tuned_model, filename)
                print('Exporting tuned: ',tuned_model)
                model_files.append(tuned_model)
                #Add list append of models here
        #Merge file lists with model list, make it into dataframe
        #Convert into pickle file, save it in the 'save_models' function
        #project_id = get_current_project(user_name)
        ops=pd.DataFrame({'model_name':models,'model_file':model_files,'fit_time':time_fit})
        #ops['fit_time']=pd.to_datetime(ops['fit_time']).dt.strftime('%d/%m/%Y %H-%M-%S')
        #print(ops['fit_time'])
        
        #model_data=pickle.dumps(ops)
        model_data=compress_pickle.dumps(ops, compression="gzip")
        file_name=current_file(user_name)
        # query = "UPDATE public.modelling SET models= %s where user_name=%s AND project_id=%s AND file_name=%s",model_data,user_name,project_id,file_name
        # sql_engine()().execute(query)
        sql_engine().execute("UPDATE public.modelling SET models= %s where user_name=%s AND project_id=%s AND file_name=%s",model_data,user_name,project_id,file_name)
        #save_models(temp_file, result.loc[:, 'Model'].to_json())
        # result= reg_result if problem_type=='regression' else class_result
        model_list = list(ml_dict.keys())
        print(model_list)
        model_list = [{'title': type(ml_dict.get(f)).__name__, 'field': f} for f in model_list] #to populate dropdown on the frontend

        if tuned_models.shape[0]>0:
            tuned_models=list(tuned_models[tuned_models['model_name'].str.startswith('Tuned')]['model_name'])
            tuned_models=[{'title': x, 'field': x} for x in tuned_models]
            print('tuned: ',tuned_models)
            model_list.extend(tuned_models)
        print(model_list)
        sort_key = 'root_mean_squared_error ' if problem_type == 'regression' else 'F1 Score '
        #print(result)
        
        order = True if problem_type == 'regression' else False
        result = result.sort_values(by=sort_key, ascending=order).reset_index(drop=True)
        print(result)
        #result.to_pickle(os.path.join(temp_dir, 'Leaderboard.pkl'))
        save_leaderboard(user_name, temp_file, result)
        validation_type = reg_validation if problem_type=='regression' else class_validation
        search_option=[{'field': f, 'title': d} for f,d in zip(list(validation_type.keys()),list(validation_type.values()))]
        transform_data = {'model_list': model_list,
                          'transform_data': result.to_html(index=False, index_names=False),
                          'search_option':search_option
                          }
        return render(request, "model_fit.html", transform_data)

    else:
        temp_file = current_file(user_name)
        project_id = get_current_project(user_name)
        problem_type = get_problem_type(user_name, temp_file)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict
        model_list = list(ml_dict.keys())
        model_list = [{'title': type(ml_dict.get(f)).__name__, 'field': f} for f in model_list]
        print(model_list[0]['title'])
        
        tuned_models=get_models(user_name, temp_file)
        if tuned_models.shape[0]>0:
            tuned_models = tuned_models[tuned_models['model_name'].str.startswith('Tuned')]
            tuned_models = [{'title': x, 'field': x} for x in tuned_models['model_name'].values]
            print(tuned_models)
            model_list.extend(tuned_models)
        print(model_list)
        result = get_leaderboard(user_name, temp_file, problem_type)
        validation_type = reg_validation if problem_type=='regression' else class_validation
        search_option=[{'field': f, 'title': d} for f,d in zip(list(validation_type.keys()),list(validation_type.values()))]
        
        model_list = {

            'model_list': model_list,
            'transform_data': result.to_html(index=False, index_names=False),
            'search_option':search_option
        }

        return render(request, "model_fit.html", model_list)


def metrics_view(request):
    user_name = request.session['user_id']
    if 'choose_model' in request.POST and request.method == "POST":
        
        print('Chosen Model: ', request.POST['premodel'])
        temp_file = current_file(user_name)
        project_id = get_current_project(user_name)
        data_in = get_current_file(user_name, temp_file)
        target = get_target(user_name, temp_file,project_id)
        columns_selected = get_columns_selected(user_name, temp_file)
        problem_type = get_problem_type(user_name, temp_file)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict
        print('Problem Type: ', problem_type)
        columns_selected = get_columns_selected(user_name, temp_file)
        print('Selected Columns: ', columns_selected)
        x = data_in.drop(target, axis=1).loc[:, columns_selected]
        y = data_in[target]
        test_size = 1/3
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=test_size / 100,
                                                                            random_state=100)
        
        
        
        models=get_models(user_name, temp_file)
        #algos=list(models['model_name'].values)
        chosen_model=request.POST['premodel']
        print(models[models['model_name']==chosen_model]['model_file'].reset_index(drop=True)[0])
        loaded_model = models[models['model_name']==chosen_model]['model_file'].reset_index(drop=True)[0]
        print('Model File :', loaded_model)
        #loaded_model = joblib.load(os.path.join(temp_dir, model_file))

        print('Model file: ', loaded_model)
        loaded_model.fit(X_train, Y_train)
        y_pred = loaded_model.predict(X_test)

        if problem_type == 'regression':
            plot_regression(loaded_model, X_train, X_test, Y_train, Y_test, y_pred)
        elif problem_type == 'classification':
            probs = loaded_model.predict_proba(X_test)
            plot_classification(loaded_model, X_train, X_test, Y_train, Y_test, y_pred, probs)
        distribution = plot()
        
        project_id = get_current_project(user_name)
        
        saved_models=get_models(user_name, temp_file)
        algos=list(models['model_name'].values)

        ml_dict = reg_dict if problem_type == 'regression' else class_dict
        model_list = []
        [model_list.append(i) for i in algos]
        model_list = [{'field': f, 'title': f} for f in model_list]
        
        distribution = {
            'distribution': distribution,
            'model_list': model_list,
            'model_chosen': type(loaded_model).__name__
        }
        return render(request, "metrics_view.html", distribution)


    else:
        temp_file = current_file(user_name)
        project_id = get_current_project(user_name)
        
        saved_models=get_models(user_name, temp_file)
        algos=list(saved_models['model_name'].values)
        problem_type = get_problem_type(user_name, temp_file)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict
        model_list = []
        [model_list.append(i) for i in algos]
        model_list = [{'field': f, 'title': f} for f in model_list]
        #model_list.extend(tuned_models)
        model_list = {

            'model_list': model_list,
        }
        return render(request, "metrics_view.html", model_list)


def interpret(request):
    user_name = request.session['user_id']
    if 'shap_model' in request.POST and request.method == "POST":

        temp_file = current_file(user_name)
        data_in = get_current_file(user_name, temp_file)
        project_id=get_current_project(user_name)
        target = get_target(user_name, temp_file,project_id)
        problem_type = get_problem_type(user_name, temp_file)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict

        print('Chosen Model: ', request.POST['premodel'])
        chosen_model = request.POST['premodel']
        
        columns_selected = get_columns_selected(user_name, temp_file)
        print('Selected Columns: ', columns_selected)
        x = data_in.drop(target, axis=1).loc[:, columns_selected]
        y = data_in[target]
        test_size = 100/3
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=test_size / 100,
                                                                            random_state=100)
        project_id = get_current_project(user_name)
        
        saved_models=get_models(user_name, temp_file)
        
        print(saved_models[saved_models['model_name']==chosen_model]['model_file'].reset_index(drop=True)[0])
        loaded_model = saved_models[saved_models['model_name']==chosen_model]['model_file'].reset_index(drop=True)[0]
        print('Model File :', loaded_model, type(loaded_model).__name__)
        
        # loaded_model.fit(X_train,Y_train)
        model_fitting = loaded_model.predict if problem_type == 'regression' else loaded_model.predict_proba
        plt.figure().clear()  # Clears background Plots
        shap.initjs()
        linear_model = ['Logistic', 'Linear']
        if any(type(loaded_model).__name__.__contains__(s) for s in linear_model):
            print('SHAP for : ', type(loaded_model).__name__)
            explainer = shap.LinearExplainer(loaded_model, masker=shap.maskers.Impute(data=X_test))
            shap_values = explainer.shap_values(X_test)
            # fig=shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, feature_names=x.columns)
            fig = shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)

        tree_models = ['Decision', 'RandomForest']
        if any(type(loaded_model).__name__.__contains__(s) for s in tree_models):
            explainer = shap.TreeExplainer(loaded_model)
            print('SHAP for : ', type(loaded_model).__name__)
            shap_values = explainer.shap_values(X_test)
            shap_values = shap_values if problem_type == 'regression' else shap_values[0]
            # fig=shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, feature_names=x.columns)
            fig = shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
        neighbour_models = ['KNeighbors', 'AdaBoost', 'SVC', 'SVR']
        if any(type(loaded_model).__name__.__contains__(s) for s in neighbour_models):
            print('SHAP for : ', type(loaded_model).__name__)
            explainer = shap.KernelExplainer(model_fitting, shap.kmeans(X_train, 10))
            shap_values = explainer.shap_values(X_test)
            shap_values = shap_values if problem_type == 'regression' else shap_values[0]
            # fig=shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, feature_names=x.columns)
            fig = shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)

        grad_models = ['XGB', 'GradientBoost']
        #if any(s.startswith() for s in grad_models):
        if any(type(loaded_model).__name__.__contains__(s) for s in grad_models):
            print('SHAP for : ', type(loaded_model).__name__)
            explainer = shap.Explainer(loaded_model, X_train)  # shap.maskers.Independent(X_train, max_samples=100))
            shap_values = explainer(X_test)
            # fig=shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, feature_names=x.columns)
            fig = shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)

        distribution = plot()

        result = get_leaderboard(user_name, temp_file, problem_type)

        distribution = {'distribution': distribution,
                        #'model_list': model_list,
                        'columns':result.columns.values,
                        'transform_data': result.round(4).to_dict(orient='records'),
                        'model': type(loaded_model).__name__
                        }
        return render(request, "interpret.html", distribution)
    
    if 'local_interpret' in request.POST and request.method == "POST":

        temp_file = current_file(user_name)
        data_in = get_current_file(user_name, temp_file)
        project_id=get_current_project(user_name)
        target = get_target(user_name, temp_file,project_id)
        problem_type = get_problem_type(user_name, temp_file)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict

        print('Chosen Model: ', request.POST['premodel'])
        chosen_model = request.POST['premodel']
        
        columns_selected = get_columns_selected(user_name, temp_file)
        print('Selected Columns: ', columns_selected)
        x = data_in.drop(target, axis=1).loc[:, columns_selected]
        y = data_in[target]
        test_size = 100/3
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=test_size / 100,
                                                                            random_state=100)

        saved_models=get_models(user_name, temp_file)
        
        print(saved_models[saved_models['model_name']==chosen_model]['model_file'].reset_index(drop=True)[0])
        loaded_model = saved_models[saved_models['model_name']==chosen_model]['model_file'].reset_index(drop=True)[0]
        n = random.randint(0, X_test.shape[0])
        test_data=X_test.iloc[n]
        actual = Y_test.iloc[n]
        predicted = loaded_model.predict(X_test.iloc[[n], :])[0]
        print(actual, ' ', predicted)
        print('Model File :', loaded_model, type(loaded_model).__name__)
        
        # loaded_model.fit(X_train,Y_train)
        model_fitting = loaded_model.predict if problem_type == 'regression' else loaded_model.predict_proba
        plt.figure().clear()  # Clears background Plots
        shap.initjs()
        linear_model = ['Logistic', 'Linear']
        if any(type(loaded_model).__name__.__contains__(s) for s in linear_model):
            print('SHAP for : ', type(loaded_model).__name__)
            explainer = shap.LinearExplainer(loaded_model, masker=shap.maskers.Impute(data=X_test))
            shap_values = explainer.shap_values(X_test)
            shap_values = shap_values if problem_type == 'regression' else shap_values[0]
            #expected_value = explainer.expected_value if problem_type == 'regression' else explainer.expected_value[0]
            distribution=shap.force_plot(explainer.expected_value, shap_values,feature_names=list(test_data.index),link='logit')
            length=(X_test.shape[1]*10)**1/3//3
            print('Plot size: ',length)
            distribution = draw_additive_plot(distribution.data, (length, 5), show=False)

        neighbour_models = ['KNeighbors', 'AdaBoost', 'SVC', 'SVR']
        if any(type(loaded_model).__name__.__contains__(s) for s in neighbour_models):
            print('SHAP for : ', type(loaded_model).__name__)
            explainer = shap.KernelExplainer(model_fitting, shap.kmeans(X_train, 10))
            shap_values = explainer.shap_values(test_data)
            shap_values = shap_values if problem_type == 'regression' else shap_values[0]
            expected_value = explainer.expected_value if problem_type == 'regression' else explainer.expected_value[0]
            distribution=shap.force_plot(expected_value, shap_values,feature_names=list(test_data.index),link='logit')
            length=(X_test.shape[1]*10)**1/3//3
            print('Plot size: ',length)
            distribution = draw_additive_plot(distribution.data, (length, 5), show=False)

        grad_models = ['XGB', 'GradientBoost']
        if any(type(loaded_model).__name__.__contains__(s) for s in grad_models):
            print('SHAP for : ', type(loaded_model).__name__)
            explainer = shap.Explainer(loaded_model, X_train)  # shap.maskers.Independent(X_train, max_samples=100))
            shap_values = explainer.shap_values(test_data)
            #shap_values = shap_values if problem_type == 'regression' else shap_values[0]
            #expected_value = explainer.expected_value if problem_type == 'regression' else explainer.expected_value[0]
            distribution = shap.force_plot(explainer.expected_value, shap_values,feature_names=list(test_data.index))#.html()
            length=(X_test.shape[1]*10)**1/3//3
            print('Plot size: ',length)
            distribution = draw_additive_plot(distribution.data, (length, 5), show=False)
        
        tree_models = ['Decision', 'RandomForest']
        if any(type(loaded_model).__name__.__contains__(s) for s in tree_models):
            explainer = shap.TreeExplainer(loaded_model)
            print('SHAP for : ', type(loaded_model).__name__)
            shap_values = explainer.shap_values(test_data)
            shap_values = shap_values if problem_type == 'regression' else shap_values[0]
            expected_value = explainer.expected_value if problem_type == 'regression' else explainer.expected_value[0]
            distribution = shap.force_plot(expected_value, shap_values,feature_names=list(test_data.index))#.html()
            length=(X_test.shape[1]*10)**1/3//3
            print('Plot size: ',length)
            distribution = draw_additive_plot(distribution.data, (length, 5), show=False)

        distribution = plot()

        #algos=list(saved_models['model_name'].values)
        #model_list = []
        #[model_list.append(i) for i in algos]
        #model_list = [{'field': f, 'title': f} for f in model_list]
        result = get_leaderboard(user_name, temp_file, problem_type)
        distribution = {'distribution': distribution,
                        #'model_list': model_list,
                        'columns':result.columns.values,
                        'transform_data': result.round(4).to_dict(orient='records'),
                        'model': loaded_model,
                        'actual': actual,
                        'predicted': predicted,
                        'test_data':pd.DataFrame(test_data).T.to_html(index=False, index_names=False)
                        }
        return render(request, "interpret.html", distribution)

    
    if 'lime_model' in request.POST and request.method == "POST":

        temp_file = current_file(user_name)
        data_in = get_current_file(user_name, temp_file)
        project_id=get_current_project(user_name)
        target = get_target(user_name, temp_file, project_id)
        problem_type = get_problem_type(user_name, temp_file)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict

        print('Chosen Model: ', request.POST['premodel'])
        chosen_model = request.POST['premodel']
        
        columns_selected = get_columns_selected(user_name, temp_file)
        print('Selected Columns: ', columns_selected)
        x = data_in.drop(target, axis=1).loc[:, columns_selected]
        y = data_in[target]
        #test_size = 100/3
        #X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=test_size / 100,random_state=100)        
        saved_models=get_models(user_name, temp_file)
        
        print(saved_models[saved_models['model_name']==chosen_model]['model_file'].reset_index(drop=True)[0])
        loaded_model = saved_models[saved_models['model_name']==chosen_model]['model_file'].reset_index(drop=True)[0]
        print('Model File :', loaded_model)

        # model=ml_dict.get(request.POST['premodel'])
        #loaded_model.fit(X_train, Y_train)
        model_fitting = loaded_model.predict if problem_type == 'regression' else loaded_model.predict_proba
        plt.figure().clear()  # Clears background Plots

        shap.initjs()

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=x.values,
            feature_names=x.columns,
            class_names=y.unique(),
            mode=problem_type,
        )
        n = random.randint(0, x.shape[0])
        test_data=x.iloc[n]
        actual = y.iloc[n]
        predicted = loaded_model.predict(x.iloc[[n], :])[0]
        print(actual, ' ', predicted)
        print('Data: ', x.iloc[[n], :])
        lime_plot = explainer.explain_instance(data_row=x.iloc[n, :], predict_fn=model_fitting).as_html()

        result = get_leaderboard(user_name, temp_file, problem_type)
        

        distribution = {'lime_plot': lime_plot,
                        'test_data':pd.DataFrame(test_data).T.to_html(index=False, index_names=False),
                        #'model_list': model_list,
                        'columns':result.columns.values,
                        'transform_data': result.round(4).to_dict(orient='records'),
                        'model': type(loaded_model).__name__,
                        'actual': actual,
                        'predicted': predicted
                        }
        return render(request, "interpret.html", distribution)
    
    if 'eli5' in request.POST and request.method == "POST":

        temp_file = current_file(user_name)
        data_in = get_current_file(user_name, temp_file)
        project_id=get_current_project(user_name)
        target = get_target(user_name, temp_file, project_id)
        problem_type = get_problem_type(user_name, temp_file)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict

        print('Chosen Model: ', request.POST['premodel'])
        chosen_model = request.POST['premodel']
        
        columns_selected = get_columns_selected(user_name, temp_file)
        print('Selected Columns: ', columns_selected)
        x = data_in.drop(target, axis=1).loc[:, columns_selected]
        y = data_in[target]
        # test_size = 100/3
        # X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=test_size / 100,random_state=100)        
        saved_models=get_models(user_name, temp_file)
        
        print(saved_models[saved_models['model_name']==chosen_model]['model_file'].reset_index(drop=True)[0])
        loaded_model = saved_models[saved_models['model_name']==chosen_model]['model_file'].reset_index(drop=True)[0]
        print('Model File :', loaded_model)
        eli5_cv = {'regression':RepeatedKFold(n_splits=5, n_repeats=100, random_state=1),
            'classification':StratifiedKFold(n_splits=5, shuffle=True)}

        perm = PermutationImportance(loaded_model, cv = eli5_cv.get(problem_type))
        perm.fit(x,y)
        eli5.show_weights(perm, feature_names = x.columns.tolist())

        # model=ml_dict.get(request.POST['premodel'])
        #loaded_model.fit(X_train, Y_train)
        # plt.figure().clear()  # Clears background Plots
        eli5_html = eli5.format_as_html(eli5.explain_weights(perm, feature_names = x.columns.tolist()))

        result = get_leaderboard(user_name, temp_file, problem_type)
        

        distribution = {'eli5_html': eli5_html,
                        # 'test_data':pd.DataFrame(test_data).T.to_html(index=False, index_names=False),
                        # 'model_list': model_list,
                        'columns':result.columns.values,
                        'transform_data': result.round(4).to_dict(orient='records'),
                        'model': type(loaded_model).__name__
                        }
        return render(request, "interpret.html", distribution)

    else:
        temp_file = current_file(user_name)
        project_id = get_current_project(user_name)
        
        saved_models=get_models(user_name, temp_file)
        algos=list(saved_models['model_name'].values)
        problem_type = get_problem_type(user_name, temp_file)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict
        #model_list = []
        #[model_list.append(i) for i in algos]
        #model_list = [{'field': f, 'title': f} for f in model_list]
        #model_list.extend(tuned_models)
        result = get_leaderboard(user_name, temp_file, problem_type)
        model_list = {

            #'model_list': model_list,
            'columns':result.columns.values,
            'transform_data': result.round(4).to_dict(orient='records'),
        }

        return render(request, "interpret.html", model_list)


def classify_metric(Y):
    # Used to detect binary/non-binary classification to choose appropriate scoring
    if len(set(Y)) <= 2:
        return 'roc_auc'
    else:
        return 'f1_weighted'
    

def tuning(request):
    user_name = request.session['user_id']
    if 'choose_model' in request.POST and request.method == "POST":
        
        project_id = get_current_project(user_name)
        temp_file = current_file(user_name)
        data_in = get_current_file(user_name, temp_file)
        problem_type = get_problem_type(user_name, temp_file)
        
        saved_models=get_models(user_name, temp_file)
        algos=list(saved_models['model_name'].values)
        folds = int(request.POST['folds'])
        
        print('Problem Type: ', problem_type)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict
        print('Chosen Model: ', request.POST['premodel'])

        target = get_target(user_name, temp_file,project_id)
        columns_selected = get_columns_selected(user_name, temp_file)
        print('Selected Columns: ', columns_selected)
        x = data_in.drop(target, axis=1).loc[:, columns_selected]
        y = data_in[target]

        chosen_model=request.POST['premodel']
        print(saved_models[saved_models['model_name']==chosen_model]['model_file'].reset_index(drop=True)[0])
        loaded_model = saved_models[saved_models['model_name']==chosen_model]['model_file'].reset_index(drop=True)[0]

        model = request.POST['premodel']
        search_type = request.POST['search_type']
        validation_type = request.POST['cross_validation']
        print('model name: ',model)
        print(search_type, ' in Progress\n')
        if request.POST['premodel'].startswith('Random'):
            print('Tuning RandomForest')
            best_params, accuracy, model_name, model_file = rf_regress(x, y, search_type,
                                               validation_type,problem_type, folds) if problem_type == 'regression' else rf_classify(x, y,
                                                                                                                 search_type,
                                                                                                                 validation_type,problem_type, folds)
        if request.POST['premodel'].startswith('XGB'):
            print('Tuning XGBClassifier')
            best_params, accuracy, model_name, model_file = xgb_regress(x, y, search_type,
                                                validation_type,problem_type, folds) if problem_type == 'regression' else xgb_classify(x,
                                                                                                                   y,
                                                                                                                   search_type,
                                                                                                                   validation_type,problem_type, folds)
        #if request.POST['premodel'] == 'log_reg':
        if request.POST['premodel'].startswith('Linear') or request.POST['premodel'].startswith('Logistic'):
            print('Tuning Logistic Regression')
            best_params, accuracy, model_name, model_file = log_regress(x, y, search_type,
                                                validation_type,problem_type, folds) if problem_type == 'regression' else log_classify(x,
                                                                                                                   y,
                                                                                                                   search_type,
                                                                                                                   validation_type,problem_type, folds)
        if request.POST['premodel'].startswith('KNeighbors'):
            print('Tuning KNNClassifier')
            best_params, accuracy, model_name, model_file = knn_regress(x, y, search_type,
                                                validation_type,problem_type, folds) if problem_type == 'regression' else knn_classify(x,
                                                                                                                   y,
                                                                                                                   search_type,
                                                                                                                   validation_type,problem_type, folds)
        if request.POST['premodel'] == 'svc':
            print('Tuning SVC')
            best_params, accuracy, model_name, model_file = svc_regress(x, y, search_type,
                                                validation_type,problem_type, folds) if problem_type == 'regression' else svc_classify(x,
                                                                                                                   y,
                                                                                                                   search_type,
                                                                                                                   validation_type,problem_type, folds)
        if request.POST['premodel'].startswith('Decision'):
            print('Tuning Decision Tree')
            best_params, accuracy, model_name, model_file = dtree_regress(x, y,
                                                  search_type,
                                                  validation_type,problem_type,folds) if problem_type == 'regression' else dtree_classify(
                x, y,
                search_type, validation_type,problem_type, folds)
        if request.POST['premodel'].startswith('Ada'):
            print('Tuning Adaboost')
            best_params, accuracy, model_name, model_file = adaboost_regress(x, y,
                                                     search_type,
                                                     validation_type,problem_type, folds) if problem_type == 'regression' else adaboost_classify(
                x, y, search_type, validation_type,problem_type, folds)
        if request.POST['premodel'].startswith('GradientBoost'):
            print('Tuning Adaboost')
            best_params, accuracy, model_name, model_file = gradboost_regress(x, y,
                                                      search_type,
                                                      validation_type,problem_type, folds) if problem_type == 'regression' else gradboost_classify(
                x, y, search_type, validation_type,problem_type, folds)
        print(best_params)
        best_params = pd.DataFrame(pd.Series(best_params)).T
        best_score = 'neg_root_mean_squared_error' if problem_type == 'regression' else classify_metric(y)
        best_params['Best Score (%s)' % best_score] = accuracy
        
        ops=pd.DataFrame({'model_name':[model_name],'model_file':[model_file]})
        
        #model_data=get_models(user_name, temp_file)
        saved_models=saved_models.drop(saved_models[saved_models['model_name']==model_name].index).append(ops).reset_index(drop="True")
        saved_models=pickle.dumps(saved_models)
        
        file_name=current_file(user_name)
        # query = "UPDATE public.modelling SET models= %s where user_name=%s AND project_id=%s AND file_name=%s",saved_models,user_name,project_id,file_name
        # engine.execute()
        sql_engine().execute("UPDATE public.modelling SET models= %s where user_name=%s AND project_id=%s AND file_name=%s",saved_models,user_name,project_id,file_name)
        saved_models=get_models(user_name, temp_file)
        algos=list(saved_models['model_name'].values)
        model_list = []
        [model_list.append(i) for i in algos]
        model_list = [{'field': f, 'title': f} for f in model_list] #Goes to frontend
        validation_type = class_split if problem_type=='classification' else reg_split
        search = class_cross_valid if problem_type=='classification' else reg_cross_valid
        search_option=[{'field': f, 'title': d} for f,d in zip(list(validation_type.keys()),search)]
        data_described = {'data_described': best_params.to_html(index=False),
                          'model_list': model_list,
                          'search_option':search_option,
                          'accuracy': accuracy}
        return render(request, "tuning.html", data_described)


    else:
        project_id = get_current_project(user_name)
        temp_file = current_file(user_name)
        problem_type = get_problem_type(user_name, temp_file)
        
        saved_models=get_models(user_name, temp_file)
        algos=list(saved_models['model_name'].values)
        print(algos)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict
        model_list = []
        [model_list.append(i) for i in algos]
        model_list = [{'field': f, 'title': f} for f in model_list]
        validation_type = reg_split if problem_type=='regression' else class_split
        search = reg_cross_valid if problem_type=='regression' else class_cross_valid
        search_option=[{'field': f, 'title': d} for f,d in zip(list(validation_type.keys()),search)]
        print(search_option)
        #model_list = [{'field': f, 'title': type(ml_dict.get(f)).__name__} for f in model_list]
        model_list = {

            'model_list': model_list,
            'search_option':search_option
        }
        return render(request, "tuning.html", model_list)


def prediction(request):
    user_name = request.session['user_id']
    if 'predict' in request.POST and request.method == "POST":
        
        temp_file = current_file(user_name)
        problem_type = get_problem_type(user_name, temp_file)
        print('Problem Type: ', problem_type)

        project_id = get_current_project(user_name)
        models=get_models(user_name, temp_file)

        chosen_model=request.POST['premodel']
        print(models[models['model_name']==chosen_model]['model_file'].reset_index(drop=True)[0])
        loaded_model = models[models['model_name']==chosen_model]['model_file'].reset_index(drop=True)[0]
        print('Model File :', loaded_model)

        
        saved_models=get_models(user_name, temp_file)
        algos=list(saved_models['model_name'].values)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict
        model_list = []
        [model_list.append(i) for i in algos]
        model_list = [{'field': f, 'title': f} for f in model_list]

        columns_selected = get_columns_selected(user_name, temp_file)
        print('Columns: ', columns_selected)
        target = get_target(user_name, temp_file,project_id)
        print('Target Column: ', target)
        test_data = get_test_file(user_name, temp_file)

        pred = loaded_model.predict(test_data.loc[:, columns_selected])
        #predicted = pd.DataFrame({'Predictions': pred})
        #predicted = pd.concat([test_data, predicted], axis=1)


        if problem_type=='classification':
            predict_probability = pd.DataFrame(loaded_model.predict_proba(test_data.loc[:, columns_selected]))
            print('Probability: ',predict_probability)
            data=predict_probability.max(axis=1)
            predict = pd.DataFrame({'Predictions': pred,'Probability':data})
            print(predict)
            predicted = pd.concat([test_data, predict], axis=1)
            #save_predictions_file(temp_file, predicted)
        else:
            predicted = pd.DataFrame({'Predictions': pred})
            predicted = pd.concat([test_data, predicted], axis=1)
            #save_predictions_file(temp_file, predicted)   

        save_predictions_file(user_name, temp_file, predicted)
        

        print('Problem Type: ', problem_type)
        if problem_type == 'regression':
            fig = px.histogram(predicted, x=predicted['Predictions'], marginal="box", hover_data=predicted.columns)
        else:
            fig = px.histogram(predicted, x=predicted['Predictions'], marginal="box", hover_data=predicted.columns,
                               color=predicted['Predictions'])
        distribution = fig.to_html(full_html=False, default_height=500, default_width=700)
        
        result = get_leaderboard(user_name, temp_file, problem_type)
        best_models=result['Model'][0]
        model_list = {
            'best_models':best_models,
            'model_list': model_list,
             #'transform_data': result.to_html(index=False, index_names=False),
            'columns':result.columns.values,
            'transform_data': result.round(4).to_dict(orient='records'),
            'distribution': distribution
        }
        return render(request, "prediction.html", model_list)
    # -----------------------------------------------------

    if 'test_file_upload' in request.POST and request.method == "POST":
        # file=request.FILES['file']
        temp_file = current_file(user_name)
        project_id=get_current_project(user_name)
        print(request.FILES.getlist("file"))
        for x in request.FILES.getlist("file"):
            file_name = str(x)
            print('File Name: ', file_name)
            start_time = time.time()
            project_id = get_current_project(user_name)
            print('Project Id: ',project_id)
            df = file_type.get(file_name.split('.')[-1])(x, na_values=' ')
            test_file_upload(user_name, temp_file, df)
            
        columns_selected = get_columns_selected(user_name, temp_file)

        # test_file_use(test_data_path)
        test_data = get_test_file(user_name, temp_file)

        target = get_target(user_name, temp_file,project_id)
        # columns_selected.extend([target])
        print(columns_selected)
        # test_data=test_data.loc[:, columns_selected]
        feat_engine = use_operation(user_name)
        print(feat_engine)
        for operation, column in zip(feat_engine['Operation'], feat_engine['Column']):
            print(operation, ' on ', column)
            print(test_data.loc[:, column].dtypes)
            print(type(operation).__name__)
            if type(operation).__name__ == 'ColumnTransformer':
                print('Operation: ', operation.transformers[0][1][0])
                try:
                    test_data.loc[:, column] = test_data.loc[:, column].apply(operation.transform)
                except:
                    test_data.loc[:, column] = operation.transform(test_data.loc[:, column])
                print(test_data.loc[:, column])
            elif type(operation).__name__ == 'function':
                print(operation.__name__)
                if operation.__name__ == 'onehotencode':
                    print(onehotencode(test_data, column))
                    test_data = onehotencode(test_data, column)
                elif operation.__name__ == 'drop_columns':
                    pass
                else:
                    print(operation)
                    print('result ', test_data.loc[:, column].apply(operation))
                    test_data.loc[:, column] = test_data.loc[:, column].apply(operation)
                    # print(operation)
                    print(test_data.loc[:, column].dtypes)
            elif type(operation).__name__ == 'defaultdict':
                type(operation)
                print('result ')
                print(test_data.loc[:, column].apply(lambda x: operation[x.name].transform(x)))
                test_data.loc[:, column] = test_data.loc[:, column].apply(lambda x: operation[x.name].transform(x))
                print(test_data.loc[:, column].dtypes)
        
        test_data = test_data.loc[:, columns_selected]
        update_test_file(user_name, temp_file, test_data)
        problem_type = get_problem_type(user_name, temp_file)
        print('Problem Type: ', problem_type)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict

        saved_models=get_models(user_name, temp_file)
        algos=list(saved_models['model_name'].values)
        model_list = []
        [model_list.append(i) for i in algos]
        model_list = [{'field': f, 'title': f} for f in model_list]

        # result = pd.read_pickle(os.path.join(temp_dir, 'Leaderboard.pkl'))
        result = get_leaderboard(user_name, temp_file, problem_type)
        best_models=result['Model'][0]
        print(test_data.head())
        model_list = {
            'best_models':best_models,
            'model_list': model_list,
            #'transform_data': result.to_html(index=False, index_names=False),
            'columns':result.columns.values,
            'transform_data': result.round(4).to_dict(orient='records'),
            'test_data': test_data.head(20).to_html(index=False, index_names=False)
        }
        return render(request, "prediction.html", model_list)


    if 'test_sql_upload' in request.POST and request.method == "POST":
        temp_file = current_file(user_name)
        project_id = get_current_project(user_name)
        hostname=request.POST.get('Hostname')
        port=str(request.POST.get('port'))
        username=str(request.POST.get('username'))
        pwd=str(request.POST.get('password'))
        database=str(request.POST.get('database'))
        table=str(request.POST.get('table'))
        query=str(request.POST.get('query'))
        print(hostname,port,username,pwd,database,query)
        print(query)
        test_engine=sqlalchemy.create_engine('postgresql://'+username+':'+pwd+'@'+hostname+':'+port+'/'+database)
        if query=="":
            data = psql.read_sql_table(table,test_engine)
        else:
            data=psql.read_sql_query(query,test_engine)
        test_file_upload(user_name, temp_file,data)
        print(data)
        
        columns_selected = get_columns_selected(user_name, temp_file)

        # test_file_use(test_data_path)
        test_data = get_test_file(user_name, temp_file)

        target = get_target(user_name, temp_file,project_id)
        # columns_selected.extend([target])
        print(columns_selected)
        # test_data=test_data.loc[:, columns_selected]
        feat_engine = use_operation(user_name)
        print(feat_engine)
        for operation, column in zip(feat_engine['Operation'], feat_engine['Column']):
            print(operation, ' on ', column)
            print(test_data.loc[:, column].dtypes)
            print(type(operation).__name__)
            if type(operation).__name__ == 'ColumnTransformer':
                print('Operation: ', operation.transformers[0][1][0])
                try:
                    test_data.loc[:, column] = test_data.loc[:, column].apply(operation.transform)
                except:
                    test_data.loc[:, column] = operation.transform(test_data.loc[:, column])
                print(test_data.loc[:, column])
            elif type(operation).__name__ == 'function':
                print(operation.__name__)
                if operation.__name__ == 'onehotencode':
                    print(onehotencode(test_data, column))
                    test_data = onehotencode(test_data, column)
                else:
                    print(operation)
                    print('result ', test_data.loc[:, column].apply(operation))
                    test_data.loc[:, column] = test_data.loc[:, column].apply(operation)
                    # print(operation)
                    print(test_data.loc[:, column].dtypes)
            elif type(operation).__name__ == 'defaultdict':
                type(operation)
                print('result ')
                print(test_data.loc[:, column].apply(lambda x: operation[x.name].transform(x)))
                test_data.loc[:, column] = test_data.loc[:, column].apply(lambda x: operation[x.name].transform(x))
                print(test_data.loc[:, column].dtypes)
        # test_data.to_csv(test_data_path, index=False)
        
        test_data = test_data.loc[:, columns_selected]
        update_test_file(user_name, temp_file, test_data)
        problem_type = get_problem_type(user_name, temp_file)
        print('Problem Type: ', problem_type)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict

        
        saved_models=get_models(user_name, temp_file)
        algos=list(saved_models['model_name'].values)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict
        model_list = []
        [model_list.append(i) for i in algos]
        model_list = [{'field': f, 'title': f} for f in model_list]

        # result = pd.read_pickle(os.path.join(temp_dir, 'Leaderboard.pkl'))
        result = get_leaderboard(user_name, temp_file, problem_type)
        best_models=result['Model'][0]
        print(test_data.head())
        model_list = {
            'best_models':best_models,
            'model_list': model_list,
            #'transform_data': result.to_html(index=False, index_names=False),
            'columns':result.columns.values,
            'transform_data': result.round(4).to_dict(orient='records'),
            'test_data': test_data.head(20).to_html(index=False, index_names=False)
        }
        return render(request, "prediction.html", model_list)


    if 'download_predictions' in request.POST and request.method == "POST":

        temp_file = current_file(user_name)
        project_id=get_current_project(user_name)
        data_in = get_predictions_file(user_name, temp_file)
        response = HttpResponse(data_in, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=Predictions.csv'
        return response

    if 'decision_tree' in request.POST and request.method == "POST":

        temp_file = current_file(user_name)
        data_in = get_current_file(user_name, temp_file)
        project_id=get_current_project(user_name)
        target = get_target(user_name, temp_file,project_id)
        columns_selected = get_columns_selected(user_name, temp_file)
        problem_type = get_problem_type(user_name, temp_file)
        print('Problem Type: ', problem_type)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict
        dtree = ml_dict.get('dt')
        features = data_in.loc[:, columns_selected]
        print(dtree)
        dtree.fit(features, data_in[target])
        
        plt.figure().clear()
        fig = plt.figure(figsize=(15, 10))
        _ = tree.plot_tree(dtree, feature_names=list(features.columns), filled=True, class_names=None, max_depth=2,
                           fontsize=15)
        tree_view = plot()
        saved_models=get_models(user_name, temp_file)
        algos=list(saved_models['model_name'].values)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict
        model_list = []
        [model_list.append(i) for i in algos]
        model_list = [{'field': f, 'title': f} for f in model_list]

        # result = pd.read_pickle(os.path.join(temp_dir, 'Leaderboard.pkl'))
        result = get_leaderboard(user_name, temp_file, problem_type)
        best_models=result['Model'][0]
        model_list = {
            'best_models':best_models,
            'model_list': model_list,
            #'transform_data': result.to_html(index=False, index_names=False),
            'columns':result.columns.values,
            'transform_data': result.round(4).to_dict(orient='records'),
            'tree_view': tree_view
        }
        return render(request, "prediction.html", model_list)
    # ------------------------------------------
    else:
        temp_file = current_file(user_name)
        problem_type = get_problem_type(user_name, temp_file)
        print('Problem Type: ', problem_type)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict
        project_id = get_current_project(user_name)
        
        saved_models=get_models(user_name, temp_file)
        algos=list(saved_models['model_name'].values)
        ml_dict = reg_dict if problem_type == 'regression' else class_dict
        model_list = []
        [model_list.append(i) for i in algos]
        model_list = [{'field': f, 'title': f} for f in model_list]
        
        result = get_leaderboard(user_name, temp_file, problem_type)
        print(result)
        if result.shape[0]>0:
            best_models=result['Model'][0]
            print('Best Models: ',best_models)
        else:
            best_models=''
        model_list = {
            'best_models':best_models,
            'model_list': model_list,
            #'transform_data': result.to_html(index=False, index_names=False),
            'columns':result.columns.values,
            'transform_data': result.round(4).to_dict(orient='records')
        }
        return render(request, "prediction.html", model_list)