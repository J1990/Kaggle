import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns

#Source: https://www.youtube.com/watch?v=Bao9GGZMLhU    
# df1 is training dataset
# df2 is non labelled dataset
def targetEncode(df1, df2, cat_name, target, weight):
    # Calculate global mean
    mean = df1[target].mean()

    # For specified column name calculate mean based on Income Value
    agg = df1.groupby(cat_name)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # calculate the "smoothed" means considering given weight
    # Overfitting incrases for lesser weights
    smooth = (counts * means + weight * mean) / (counts + weight)

    # Replace each value by the according smoothed mean
    if df2 is None:
        return df1[cat_name].map(smooth)
    else:
        return df1[cat_name].map(smooth),df2[cat_name].map(smooth.to_dict())

def main():
    dataset = pd.read_csv(r'D:\OneDrive\MSc\Sem 1\Machine Learning\KaggleIndividualCompetition\Code\trainingData.csv')
    nonLabelledDataset = pd.read_csv(r'D:\OneDrive\MSc\Sem 1\Machine Learning\KaggleIndividualCompetition\Code\dataWithoutLabels.csv')
    dataset = dataset.fillna(method='ffill')
    nonLabelledDataset = nonLabelledDataset.fillna(method='ffill')

    weight = 3
    dataset['Gender'], nonLabelledDataset['Gender'] = targetEncode(dataset, nonLabelledDataset, 'Gender', 'Income in EUR', weight)
    dataset['Profession'], nonLabelledDataset['Profession'] = targetEncode(dataset, nonLabelledDataset, 'Profession', 'Income in EUR', weight)
    dataset['University Degree'], nonLabelledDataset['University Degree'] = targetEncode(dataset, nonLabelledDataset, 'University Degree', 'Income in EUR', weight)
    dataset['Country'], nonLabelledDataset['Country'] = targetEncode(dataset, nonLabelledDataset, 'Country', 'Income in EUR', weight)

    #Convert NAN values for 'Country': 0 & 'Profession': 0
    nonLabelledDataset = nonLabelledDataset.fillna(0)

    #Values to predict
    y = dataset['Income in EUR'].values
    
    #Find correlation between features
    #num_features = ['Year of Record','Age','Size of City','Wears Glasses','Body Height [cm]','Gender','Country','Profession', 'University Degree', 'Hair Color', 'Income in EUR']
    #g = sns.heatmap(dataset[num_features].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
    #Remove features with negligible correlation
    X = dataset.drop(['Income in EUR','Instance','Hair Color','Wears Glasses'], axis='columns').values
    
    #Divide Training and Test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
    
    #print(nonLabelledDataset.isnull().sum())

    #regressor = LinearRegression()
    #regressor = XGBClassifier()
    regressor= RandomForestRegressor(n_estimators=1000,random_state=42)
    #regressor = GradientBoostingRegressor(n_estimators=1000, learning_rate = 0.001, random_state = 10)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    print("Root mean squared error: %.2f"
      % np.sqrt(mean_squared_error(y_test, y_pred)))    

    X = nonLabelledDataset.drop(['Income','Instance','Hair Color','Wears Glasses'], axis='columns').values
    predictedData = regressor.predict(X)

    df = pd.DataFrame({'Instance': nonLabelledDataset['Instance'], 'Predicted': predictedData})

    fileName = r'D:\OneDrive\MSc\Sem 1\Machine Learning\KaggleIndividualCompetition\Code\submissionData.csv'
    df.to_csv(fileName, sep=',')

    print('done')

if __name__ == '__main__':
  main()