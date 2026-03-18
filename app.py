
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

st.set_page_config(page_title="Universal Bank Marketing AI", layout="wide")

st.title("Universal Bank – Personal Loan Marketing Intelligence Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("UniversalBank.csv")

df = load_data()

tabs = st.tabs(["Overview","Customer Insights","Financial Behaviour","Model Performance","Prediction Tool"])

with tabs[0]:
    st.subheader("Executive Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Customers", df.shape[0])
    c2.metric("Average Income ($000)", round(df["Income"].mean(),2))
    c3.metric("Avg Credit Card Spend ($000)", round(df["CCAvg"].mean(),2))
    c4.metric("Loan Acceptance Rate", str(round(df["Personal Loan"].mean()*100,2))+"%")
    st.dataframe(df.head())

with tabs[1]:
    col1,col2 = st.columns(2)
    fig = px.histogram(df, x="Age", nbins=30, template="plotly_white", title="Customer Age Distribution")
    col1.plotly_chart(fig, use_container_width=True)
    fig = px.pie(df, names="Personal Loan", template="plotly_white", title="Loan Acceptance Distribution")
    col2.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    col1,col2 = st.columns(2)
    fig = px.histogram(df, x="Income", nbins=40, template="plotly_white", title="Income Distribution")
    col1.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(df, x="CCAvg", nbins=40, template="plotly_white", title="Credit Card Spending")
    col2.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    X = df.drop(columns=["Personal Loan","ID"])
    y = df["Personal Loan"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
    models={
    "Decision Tree":DecisionTreeClassifier(),
    "Random Forest":RandomForestClassifier(),
    "Gradient Boosting":GradientBoostingClassifier()
    }
    results=[]
    roc_fig=plt.figure()
    for name,model in models.items():
        model.fit(X_train,y_train)
        train_pred=model.predict(X_train)
        test_pred=model.predict(X_test)
        train_acc=accuracy_score(y_train,train_pred)
        test_acc=accuracy_score(y_test,test_pred)
        precision=precision_score(y_test,test_pred)
        recall=recall_score(y_test,test_pred)
        f1=f1_score(y_test,test_pred)
        results.append([name,train_acc,test_acc,precision,recall,f1])
        probs=model.predict_proba(X_test)[:,1]
        fpr,tpr,_=roc_curve(y_test,probs)
        roc_auc=auc(fpr,tpr)
        plt.plot(fpr,tpr,label=name+" AUC="+str(round(roc_auc,3)))
        cm=confusion_matrix(y_test,test_pred)
        fig,ax=plt.subplots()
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
        st.pyplot(fig)
    results_df=pd.DataFrame(results,columns=["Model","Train Accuracy","Test Accuracy","Precision","Recall","F1 Score"])
    st.dataframe(results_df)
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    st.pyplot(roc_fig)

with tabs[4]:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        new_data=pd.read_csv(uploaded)
        model=RandomForestClassifier()
        model.fit(X,y)
        preds=model.predict(new_data)
        new_data["Predicted Personal Loan"]=preds
        st.dataframe(new_data.head())
        csv=new_data.to_csv(index=False).encode()
        st.download_button("Download Results",csv,"loan_predictions.csv","text/csv")
