import pandas as pd
from sklearn.decomposition import PCA


if __name__=="__main__":
    fin='/Users/skyeong/pythonwork/creditcard_fraud/data/creditcard.csv'
    df=pd.read_csv(fin)
    # df  = pd.read_csv(fin,usecols=colnames)
    df1 = df.iloc[:,1:28]

    X = df1.values
    pca = PCA(n_components=2)
    pca.fit(X.T)
    # PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,svd_solver='auto', tol=0.0, whiten=False)

    print('singular value :', pca.singular_values_)
    print('singular vector :\n', pca.components_.T)
    print('eigen_value :', pca.explained_variance_)
    print('explained variance ratio :', pca.explained_variance_ratio_)

