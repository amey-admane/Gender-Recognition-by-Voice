
from django.contrib import admin
from django.urls import path,include
from .views import Gradient_Model,Ensemble_Model,Comparision,univariate_bivariate,home,SVM_Model,Random_Forest_Model,Decision_Model,KNN_Model,Logistic_Regreesion_Model
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('Explore_Data/', univariate_bivariate,name= "Explore_Data"),
    path('Gradient_Model/', Gradient_Model,name= "Gradient_Model"),
    path('SVM_Model/', SVM_Model,name= "SVM_Model"),
    path('Random_Forest_Model/', Random_Forest_Model,name= "Random_Forest_Model"),
    path('Decision_Model/', Decision_Model,name= "Decision_Model"),
    path('KNN_Model/', KNN_Model,name= "KNN_Model"),
    path('Ensemble_Model/', Ensemble_Model,name= "Ensemble_Model"),
    path('Comparision/', Comparision,name= "Comparision"),
    path('Logistic_Regreesion_Model/', Logistic_Regreesion_Model,name= "Logistic_Regreesion_Model"),
    path('', home,name= ""),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
