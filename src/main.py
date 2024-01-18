import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import RandomizedSearchCV
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,confusion_matrix,roc_auc_score,roc_curve,precision_recall_curve

def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    return df

data = load_data()

num_df = data.select_dtypes(include=['int','float'])
cat_df = data.select_dtypes(exclude=['int','float'])

columns_to_transform_2 = ['purchase_lead','length_of_stay','flight_hour','flight_duration']
data[columns_to_transform_2] = data[columns_to_transform_2].apply(lambda x: np.log1p(x))

standard_columns = ['flight_hour','flight_duration']
robust_columns = ['purchase_lead','length_of_stay']

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

data[standard_columns] = std_scaler.fit_transform(data[standard_columns])
data[robust_columns] = rob_scaler.fit_transform(data[robust_columns])

X = data.drop(columns=['booking_complete'])
y = data['booking_complete']

target_enc = TargetEncoder()
cat_target_enc = target_enc.fit_transform(cat_df,y)

X.drop(columns=['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin'],inplace=True)
X = pd.concat([X,cat_target_enc],axis=1)

smote = SMOTE(sampling_strategy='auto',random_state=42,n_jobs=-1)

rf_model_4 = make_imb_pipeline(smote, RandomForestClassifier(n_estimators=100, random_state=42))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scores = cross_validate(rf_model_4,X_train,y_train,cv=5,scoring=['precision','accuracy','recall',])

rf_model_4.fit(X_train,y_train)
y_pred_4 = rf_model_4.predict(X_test)


param_grid = {
    'randomforestclassifier__criterion': ['gini', 'entropy'],
    'randomforestclassifier__max_depth': [2, 4, 6, 10, 20],
    'randomforestclassifier__max_features': ['sqrt', 'log2', None],
    'smote__sampling_strategy': ['auto', 0.5, 0.75, 1.0],
    'randomforestclassifier__min_samples_leaf': [None, 2, 4, 6],
    'randomforestclassifier__min_samples_split': [1, 2, 4, 6, 8, 10],
    'randomforestclassifier__n_estimators': [50, 80, 100, 120, 150]
}

scoring = {'accuracy': 'accuracy', 'auc': 'roc_auc'}

random_search = RandomizedSearchCV(
    rf_model_4,
    param_distributions=param_grid,
    n_iter=10,
    scoring=scoring,
    refit='auc',
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train,y_train)

hyper_smote = SMOTE(sampling_strategy=0.75,random_state=42,n_jobs=-1)

hyperparam_model = make_imb_pipeline(hyper_smote, RandomForestClassifier(n_estimators=150,
                                                             min_samples_split=8,
                                                             min_samples_leaf=6,
                                                             max_features='sqrt',
                                                             max_depth=20, 
                                                             random_state=42))

hyperparam_model.fit(X_train,y_train)

y_pred_param = hyperparam_model.predict(X_test)

classifier = hyperparam_model.named_steps['randomforestclassifier']
param_feature_importances = classifier.feature_importances_

importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': param_feature_importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)