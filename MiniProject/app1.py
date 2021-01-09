import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,accuracy_score, roc_auc_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Default of Credit Card Clients")
    st.sidebar.title("Default of Credit Card Clients")

    df = pd.read_csv('cleaned_data.csv')

    @st.cache(persist=True)
    def splitting_the_data(df):
        items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university']
        features_response = df.columns.tolist()
        features_response = [item for item in features_response if item not in items_to_remove]

        X=df[features_response[:-1]]
        y=df['default payment next month']

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=24)
        return X_train,X_test,y_train,y_test


    def plotgraph(curve,X_test,y_test):
        if 'Confusion Matrix' in curve:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(rf,X_test,y_test,display_labels=['Default','Not Default'])
            st.pyplot()

        if 'ROC Curve' in curve:
            st.subheader("ROC Curve")
            plot_roc_curve(rf,X_test,y_test)
            st.pyplot()
            # fig = px.line(df, x=X_test, y=y_test, title='Life expectancy in Canada')
            # fig.show()


        if 'Precision Recall Curve' in curve:
            st.subheader("Precision Recall Curve")
            plot_precision_recall_curve(rf,X_test,y_test)
            st.pyplot()


    st.sidebar.subheader('Random Forest Classifier')
    st.sidebar.subheader('-----------------------------')
    st.sidebar.subheader("Choose Hyperparameters")
    n_estimators = st.sidebar.number_input("The number of trees in the forest",100,5000,step=10,key='n_est')
    max_depth = st.sidebar.number_input("The maximum depth of the tree",1,20,2,step=1,key='max_depth')
    curve = st.sidebar.selectbox("Which Curve to plot?",('ROC Curve','Precision Recall Curve','Confusion Matrix'),key='1')

    if st.sidebar.button("RUN",key='class'):
        st.subheader("Random Forest Result")
        X_train,X_test,y_train,y_test = splitting_the_data(df)
        rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
        rf.fit(X_train, y_train)
        accuracy = rf.score(X_test,y_test)
        y_pred = rf.predict(X_test)
        
        st.write("ROC AUC Score:",roc_auc_score(y_test, y_pred).round(2))
        st.write("Accuracy:",accuracy_score(y_test,y_pred).round(2))
        st.write("Precision:",precision_score(y_test,y_pred).round(2))
        st.write("Recall:",recall_score(y_test,y_pred).round(2))
        plotgraph(curve,X_test,y_test)



if __name__ == '__main__':
    main()