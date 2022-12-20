from django.shortcuts import render

# Create your views here.
import io
import urllib,base64
import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, cohen_kappa_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 
from templates import *
from django.views.decorators.csrf import csrf_exempt
from .temp import *
import csv
from statistics import mode
from sklearn import metrics 
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_curve, auc


from sklearn.linear_model import LogisticRegression  

from yellowbrick.classifier import ClassificationReport

from sklearn.linear_model import LogisticRegression  

from sklearn.ensemble import RandomForestClassifier

features = {
    'meanfreq',
    'sd', 
    'median', 
    'Q25', 
    'Q75', 
    'IQR', 
    'skew', 
    'kurt',
    'sp.ent', 
    'sfm', 
    'mode', 
    'centroid', 
    'meanfun', 
    'minfun', 
    'maxfun',
    'meandom', 
    'mindom', 
    'maxdom', 
    'dfrange', 
    'modindx', 
    'label',
}

@csrf_exempt
def Comparision(request):
    return render(request,'Compare.html')

def home(request):
    return render(request,'main.html')

@csrf_exempt
def univariate_bivariate(request):
    df = pd.read_csv("voice.csv")
    context = {'options' :features,
        'model_info':'''In Exploratory Dataset Analysis ,we have done univariate and bivariate analysis.Using univariate analysis,user will get the idea about how the data is distributed
        Using Bivariate analysis,user will get the idea about how 2 features are related.Results from this analysis, will help the user to select best features for training the model.
        '''}
    if request.method=="GET":
        print("in get")
        context["show"] ="donotshow"
        context['type'] = "Exploratory Dataset Analysis"
    else:
        if request.method=="POST" and "customCheck1" in request.POST:
            print(request)
            print(request.POST.getlist('customCheck1'))
            feature = request.POST.getlist('customCheck1')
        
            if len(feature)==1:
                print("in 1")
                palette = ["#f37852", "#7d1f5a"]
                axes=sb.displot(x =feature[0],kde=False,bins = 15, hue = df['label'] , palette = palette, data=df)
                sb.set(rc={'figure.facecolor':'#def2f1','axes.facecolor':'#def2f1'})
                plt.grid(False)
                fig = plt.gcf()
                #convert graph into dtring buffer and then we convert 64 bit code into image
                buf = io.BytesIO()
                fig.savefig(buf,format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                uri1 =  urllib.parse.quote(string)
                context['img1'] = uri1
                context['type'] = "Univariate"
                context["feature_a"] = feature[0]

      
            if len(feature)==2:

                print("in 2")
                fig = plt.figure(figsize=(10, 4))
                sb.scatterplot(x = feature[0], y =  feature[1], hue ='label',data = df)
                plt.grid(False)
                fig = plt.gcf()
                #convert graph into dtring buffer and then we convert 64 bit code into image
                buf = io.BytesIO()
                fig.savefig(buf,format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                uri2 =  urllib.parse.quote(string)
                context['img1'] = uri2
                context['type'] = "Bivariate"

                context["feature_a"] = feature[0]
                context["feature_b"] = feature[1]

    return render(request,'Explore_Page.html',context)







@csrf_exempt
def Gradient_Model(request):
    context = {'options' :features}
    if request.method == "POST":
  
        features_list   = request.POST.getlist('customCheck1')
        if features_list[0] == "Select All" :
            features_list = features.copy()

        features_to_use = features-set(features_list)
        features_to_use.add('label')
        
        df = pd.read_csv("voice.csv")
        data = df.copy()
        
        df['label']  = [1 if i == "male" else 0 for i in df.label]
        
       
        list = []
        print(len(features_to_use))
        y = df.label.values
        x = data.drop(features_to_use,axis=1)
       

        x_train, x_test, y_train, y_test = train_test_split (x,y,test_size=0.2,random_state = 50)

        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
        gb.fit(x_train, y_train)
       

        predications = gb.predict(x_test)


        context = model_footer(x_test, y_test,predications)

        context['model_name'] = 'Gradient Boosting'

        context['options'] = features

        context['model_info'] = '''Gradient Boosting is a popular boosting algorithm. In gradient boosting, each predictor corrects its predecessor’s error. In contrast to Adaboost, the weights of the training instances are not tweaked, instead, each predictor is trained using the residual errors of predecessor as labels.

There is a technique called the Gradient Boosted Trees whose base learner is CART (Classification and Regression Trees).

The below diagram explains how gradient boosted trees are trained for regression problems.'''

        context["show"] = "show"
        # context={
        #     'list_acc' :list,
         
        #     'data1':uril,
        #     'data2':uril2
        # }
        return render(request,"index.html",context)
    else:
        context["show"] = "donotshow"
        context['model_info'] = '''Gradient Boosting is a popular boosting algorithm. In gradient boosting, each predictor corrects its predecessor’s error. In contrast to Adaboost, the weights of the training instances are not tweaked, instead, each predictor is trained using the residual errors of predecessor as labels.

There is a technique called the Gradient Boosted Trees whose base learner is CART (Classification and Regression Trees).

The below diagram explains how gradient boosted trees are trained for regression problems.'''

        context['model_name'] = 'Gradient Boosting'
        return render(request,"index.html",context)




def model_start_scale(features_to_use):
    df = pd.read_csv('voice.csv')
    y = df.label

    data = df.copy()
        
    df['label']  = [1 if i == "male" else 0 for i in df.label]
    
   
    list = []
    print(len(features_to_use))
    y = df.label.values
    x = data.drop(features_to_use,axis=1)


    # scaling 

    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    return x,y



def model_start(features_to_use):
    df = pd.read_csv('voice.csv')
    y = df.label

    data = df.copy()
        
    df['label']  = [1 if i == "male" else 0 for i in df.label]
    
   
    list = []
    print(len(features_to_use))
    y = df.label.values
    x = data.drop(features_to_use,axis=1)

   
    return x,y








def model_footer(x_test,y_test,predications):
    
    list =[]

    accscore = accuracy_score (y_test, predications)
    recscore = recall_score (y_test, predications)
    f1score = f1_score (y_test, predications)
    kappascore =cohen_kappa_score(y_test,predications)
    prescor = precision_score(y_test, predications)
    list.append(accscore)
    list.append(prescor)
    list.append(recscore)
    list.append(f1score)

    list_name = ["Accuracy score :","Precision score :","Recall score :","F1 score :"]
   
    classes = ['correct','incorrect']
   
    total_samples = x_test.shape[0]
    correct_samples = metrics.accuracy_score(y_test, predications,normalize = False)
    incorrect_samples = total_samples - correct_samples
   
    # pie chart
    lab = ["Correct", "Incorrect"]
    piemf = np.array([correct_samples,incorrect_samples])
    explode = (0.1, 0.0)
    colors = ("#f37852", "#7d1f5a")
    wp = { 'linewidth' : 1, 'edgecolor' : "white" }
    def func(pct, allvalues):
            absolute = int(pct / 100.*np.sum(allvalues))
            return "{:.1f}%".format(pct)
        
    fig, ax = plt.subplots(figsize =(4, 4.2))
    wedges, texts, autotexts = ax.pie(piemf,
        autopct = lambda pct: func(pct, piemf),
        explode = explode,
        labels = lab,
        shadow = True,
        colors = colors,
        startangle = 50,
        wedgeprops = wp,
        textprops = dict(color = "#17252a")
    )


    ax.legend(wedges, lab,
            title ="Samples",
            bbox_to_anchor=(0., 1.02, 1., .102),
            loc='lower left',
            ncol=2, mode="expand", borderaxespad=0.
            )
    plt.setp(autotexts, size = 15, weight ="bold")

    fig = plt.gcf()
    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string2 = base64.b64encode(buf.read())
    uril =  urllib.parse.quote(string2)




    # Roc curve
    fpr1, tpr1, thresholds = metrics.roc_curve(y_test, predications, pos_label=0)
    plt.subplot(211)
    # plt.ylabel("True Positive Rate")
    # plt.xlabel("False Positive Rate")
    plt.title("ROC Curve")
    
    
    plt.plot(tpr1,fpr1)
    auc = np.trapz(fpr1,tpr1)
    print("Area Under ROC Curve:", auc)

    
    # plt.show()
    fig_2 = plt.gcf()

    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf_2 = io.BytesIO()
    fig_2.savefig(buf_2,format='png')
    buf_2.seek(0)
    string1 = base64.b64encode(buf_2.read())
    uril2 =  urllib.parse.quote(string1)
    

    context={
        'list_acc' :list,
        'score1':round(accscore,4),
        'score2':round(prescor,4),
        'score3':round(recscore,4),
        'score4':round(f1score,4),
        'data1':uril,
        'data2':uril2,
        "auc_value" : round(auc,4),
    }

    return context




@csrf_exempt
def SVM_Model(request):
    context = {'options' :features}
    if request.method == "POST":
  
        features_list   = request.POST.getlist('customCheck1')
        if features_list[0] == "Select All" :
            features_list = features.copy()

        
        features_to_use = features-set(features_list)
        features_to_use.add('label')
        
        x, y = model_start_scale(features_to_use)
        x_train, x_test, y_train, y_test = train_test_split (x,y,test_size=0.2,random_state = 50)

        gb = SVC(kernel='rbf', gamma=0.01)
        gb.fit(x_train, y_train)
       

        predications = gb.predict(x_test)


        context = model_footer(x_test, y_test,predications)

        context['model_name'] = 'Support Vector Machine'

        context['options'] = features

        context['model_info'] = '''SVM is supervised machine learning model used for classification and regression. The main objective of SVM is to find a
         hyperplane in a N-dimensional space that distinctly classifies the data points of the dataset. 
         A hyperplane is a decision boundary.  SVM uses kernel functions to systematically find support vector
          classifier in higher dimensions. We can implement the SVM using various kernels, some of which are linear, polynomial, rbf, gussian, sigmoid,etc.'''

        context["show"] = "show"

        return render(request,"index.html",context)
    else:
        context['model_info'] = '''SVM is supervised machine learning model used for classification and regression. The main objective of SVM is to find a
         hyperplane in a N-dimensional space that distinctly classifies the data points of the dataset. 
         A hyperplane is a decision boundary.  SVM uses kernel functions to systematically find support vector
          classifier in higher dimensions. We can implement the SVM using various kernels, some of which are linear, polynomial, rbf, gussian, sigmoid,etc.'''

        context["show"] = "donotshow"
        context['model_name'] = 'Support Vector Machine'

        return render(request,"index.html",context)



@csrf_exempt
def Random_Forest_Model(request):
    context = {'options' :features}
    if request.method == "POST":
  
        features_list   = request.POST.getlist('customCheck1')
        if features_list[0] == "Select All" :
            features_list = features.copy()

        features_to_use = features-set(features_list)
        features_to_use.add('label')
               
        x, y = model_start_scale(features_to_use)
        x_train, x_test, y_train, y_test = train_test_split (x,y,test_size=0.2,random_state = 50)

        gb = SVC(kernel='rbf', gamma=0.01)
        gb.fit(x_train, y_train)
       

        predications = gb.predict(x_test)


        context = model_footer(x_test, y_test,predications)

        context['model_name'] = 'Random Forest'

        context['options'] = features

        context['model_info'] = '''Random forest is a Supervised and Ensemble Machine Learning 
        Algorithm that is used widely in Classification and Regression problems. It builds decision trees 
        on different samples and takes their
         majority vote for classification and average in case of regression.'''

        context["show"] = "show"
        return render(request,"index.html",context)
    else:
        context['model_info'] = '''Random forest is a Supervised and Ensemble Machine Learning 
        Algorithm that is used widely in Classification and Regression problems. It builds decision trees 
        on different samples and takes their
         majority vote for classification and average in case of regression.'''

        context["show"] = "donotshow"
        context['model_name'] = 'Random Forest'

        return render(request,"index.html",context)





@csrf_exempt
def Logistic_Regreesion_Model(request):
    context = {'options' :features}
    if request.method == "POST":
  
        features_list   = request.POST.getlist('customCheck1')
        if features_list[0] == "Select All" :
            features_list = features.copy()

        features_to_use = features-set(features_list)
        features_to_use.add('label')
               
        x, y = model_start_scale(features_to_use)
        x_train, x_test, y_train, y_test = train_test_split (x,y,test_size=0.2,random_state = 50)

        gb = LogisticRegression(C=10.0,max_iter= 100)
        gb.fit(x_train, y_train)
       

        predications = gb.predict(x_test)


        context = model_footer(x_test, y_test,predications)

        context['model_name'] = 'Logistic Regression'

        context['options'] = features

        context['model_info'] = '''Logistic Regression is a supervised machine learning model used for
         prediction of probability of target variable. It finds the probability between dependent and 
         independent variables using the logistic regression equation.
         It is used when we have categorical target variables.'''

        context["show"] = "show"

        return render(request,"index.html",context)
    else:
        context['model_info'] = '''Logistic Regression is a supervised machine learning model used for
         prediction of probability of target variable. It finds the probability between dependent and 
         independent variables using the logistic regression equation.
         It is used when we have categorical target variables.'''
        context["show"] = "donotshow"
        context['model_name'] = 'Logistic Regression'
        return render(request,"index.html",context)





@csrf_exempt
def KNN_Model(request):
    context = {'options' :features}
    context['model_info'] = '''K-nearest neighbors (KNN) is a type of supervised learning algorithm used for both regression and classification. KNN tries to predict the correct class for the test data by calculating the distance between the test data and all the training points. Then select the K number of points which is closet to the test data.
'''

    if request.method == "POST":
  
        features_list   = request.POST.getlist('customCheck1')
        if features_list[0] == "Select All" :
            features_list = features.copy()

        features_to_use = features-set(features_list)
        features_to_use.add('label')
               
        x, y = model_start_scale(features_to_use)
        x_train, x_test, y_train, y_test = train_test_split (x,y,test_size=0.2,random_state = 50)

        gb = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform',algorithm = 'brute',metric = 'minkowski')
        gb.fit(x_train, y_train)
       

        predications = gb.predict(x_test)


        context = model_footer(x_test, y_test,predications)

        context['model_name'] = 'K Nearest Neighbour '

        context['options'] = features

        
        context["show"] = "show"

        return render(request,"index.html",context)
    else:

        context["show"] = "donotshow"
        context['model_name'] = 'K Nearest Neighbour '

        return render(request,"index.html",context)






@csrf_exempt
def Decision_Model(request):
    context = {'options' :features}
    if request.method == "POST":
  
        features_list   = request.POST.getlist('customCheck1')
        if features_list[0] == "Select All" :
            features_list = features.copy()

        features_to_use = features-set(features_list)
        features_to_use.add('label')
               
        x, y = model_start(features_to_use)
        x_train, x_test, y_train, y_test = train_test_split (x,y,test_size=0.2,random_state = 50)

        gb = DecisionTreeClassifier(criterion = "gini",random_state = 30,max_depth=5, min_samples_leaf=4)
        gb.fit(x_train, y_train)
       

        predications = gb.predict(x_test)


        context = model_footer(x_test, y_test,predications)

        context['model_name'] = 'Decision Tree '

        context['options'] = features

        context['model_info'] = '''A decision tree has a tree like structure, in which every internal node represents a feature,
         branch represents a decision and leaf nodes represent the final outcome. By using ID3 algorithm, 
         a node is split into two or more sub-nodes. It works by selecting the feature with maximum information 
         gain. There are 2 different criteria for the selection of information gain - Gini index and Entropy.'''

        context["show"] = "show"

        return render(request,"index.html",context)
    else:
        context['model_info'] = '''A decision tree has a tree like structure, in which every internal node represents a feature,
         branch represents a decision and leaf nodes represent the final outcome. By using ID3 algorithm, 
         a node is split into two or more sub-nodes. It works by selecting the feature with maximum information 
         gain. There are 2 different criteria for the selection of information gain - Gini index and Entropy.'''

        context["show"] = "donotshow"
        context['model_name'] = 'Decision Tree '
        return render(request,"index.html",context)




@csrf_exempt
def Ensemble_Model(request):
    print("In se")
    context = {'options' :features}
    if request.method == "POST":
  
        features_list   = request.POST.getlist('customCheck1')
        if features_list[0] == "Select All" :
            features_list = features.copy()


        features_to_use = features-set(features_list)
        features_to_use.add('label')
               
        x, y = model_start_scale(features_to_use)
        x_train, x_test, y_train, y_test = train_test_split (x,y,test_size=0.2,random_state = 50)

        
        main_model =LogisticRegression()
        #model2 =KNeighborsClassifier(n_neighbors=10)
        model2 = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform',algorithm = 'brute',metric = 'minkowski')
        model3= SVC()


        main_model.fit(x_train,y_train)
        model2.fit(x_train,y_train)
        model3.fit(x_train,y_train)



        pred1=main_model.predict(x_test)
        pred2=model2.predict(x_test)
        pred3=model3.predict(x_test)


        final_pred = np.array([])
        for i in range(0,len(x_test)):
            final_pred = np.append(final_pred, mode([pred1[i],pred2[i],pred3[i]]))
        int_array = final_pred. astype(int)

        cnt=0

        for i in range(0,len(x_test)):
            if final_pred[i]!=y_test[i]:
                cnt+=1
        ans = ((len(x_test)-cnt)/len(x_test))

        print('Accuracy Score:   ',end=" ")
        print(metrics.accuracy_score(y_test,final_pred))
        print("Out of total "+str(len(x_test))+" samples "+str(metrics.accuracy_score(y_test, final_pred,normalize = False))+" samples were predicted correctly")


        accscore = accuracy_score (y_test, final_pred)
        recscore = recall_score (y_test, final_pred)
        f1score = f1_score (y_test, final_pred)
        prescor = precision_score(y_test, final_pred)

        # context = model_footer(x_test, y_test,predications)
        context={
            'score1':round(accscore,4),
            'score2':round(prescor,4),
            'score3':round(recscore,4),
            'score4':round(f1score,4),
        }
        
        context['model_name'] = 'Ensemble Learning'

        context['options'] = features

        context['model_info'] = '''Ensemble model is build using Support Vector Machine
        ,K Nearest Neighbours and Logistic Regression. All the three models will be predicting the 
        output and using Mode,the most common output will become the final output.'''
        context["show"] = "show"

        return render(request,"ensemble.html",context)
    else:

        context["show"] = "donotshow"
        context['model_info'] = '''Ensemble model is build using Support Vector Machine
        ,K Nearest Neighbours and Logistic Regression. All the three models will be predicting the 
        output and using Mode,the most common output will become the final output.'''
        
        context['model_name'] = 'Ensemble Learning'
        return render(request,"ensemble.html",context)

