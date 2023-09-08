from django.urls import path
from .views import home, SolverML, login_page, register_page, datainput, dataview, transform, visual, modelling,ml_model, metrics_view, interpret, tuning, prediction
#, , regression, clustering
from django.conf.urls.static import static
from django.conf import settings
urlpatterns = [
    path('homepage/',home, name='homepage'),
    path('SolverML/', SolverML, name='SolverML'),
    path('login/', login_page, name='login_page'),
    path('register/', register_page, name='register_page'),
    path('homepage/datainput/', datainput, name='datainput'),
    path('homepage/dataview/', dataview, name='dataview'),
    path('homepage/transform/', transform, name='transform'),
    path('homepage/visual/', visual, name='visual'),
    path('homepage/modelling/', modelling, name='modelling'),
    path('homepage/model_fit/',ml_model,name='model_fit'),
    path('homepage/metrics_view/',metrics_view,name='metrics_view'),
    path('homepage/interpret/',interpret,name='interpret'),
    path('homepage/tuning/',tuning,name='tuning'),
    path('homepage/prediction/',prediction,name='prediction')
]