import pandas as pd


df = pd.read_csv("higgs_boson_data.csv")


import numpy as np
df.replace(to_replace=-999.000,value=np.nan,inplace=True)


df['DER_mass_MMC'].replace(to_replace=np.nan,value=df['DER_mass_MMC'].mean(),inplace=True)

# This variable is highly correlated with DER_prodeta_jet_jet and should be ignored for analysis (Correlation = 0.9999)
df.drop(['DER_lep_eta_centrality'],inplace=True,axis = 1)

# This variable is highly correlated with DER_deltaeta_jet_jet and should be ignored for analysis (Correlation = 0.94604)
df.drop(['DER_mass_jet_jet'],inplace=True,axis =1)

# This variable is highly correlated with DER_mass_jet_jet and should be ignored for analysis     (Correlation= 0.94444)
df.drop(['DER_prodeta_jet_jet'],inplace=True,axis =1)

# This variable is highly correlated with PRI_jet_leading_pt and should be ignored for analysis   (Correlation = 0.9961)
df.drop(['PRI_jet_leading_eta'],inplace=True,axis =1)

# This variable is highly correlated with PRI_jet_leading_eta and should be ignored for analysis   (Correlation = 0.9999)
df.drop(['PRI_jet_leading_phi'],inplace=True,axis =1)

df['PRI_jet_leading_pt'].replace(to_replace=np.nan,value=df['PRI_jet_leading_pt'].mean(),inplace=True)

# This variable is highly correlated with PRI_jet_subleading_pt and should be ignored for analysis (Correlation = 0.99935)
df.drop(['PRI_jet_subleading_eta'],inplace=True,axis =1)

# This variable is highly correlated with PRI_jet_subleading_eta and should be ignored for analysis (Correlation = 0.99999)
df.drop(['PRI_jet_subleading_phi'],inplace=True,axis =1)

df['PRI_jet_subleading_pt'].replace(to_replace=np.nan,value=df['PRI_jet_subleading_pt'].mean(),inplace=True)

# This variable is highly correlated with PRI_jet_subleading_eta and should be ignored for analysis (Correlation = 0.94604)
df.drop(['DER_deltaeta_jet_jet'],inplace=True,axis =1)






import sklearn.utils
df = sklearn.utils.shuffle(df)
X = df.iloc[:,1:24].astype(float)
Y = df.iloc[:,24:]



#from sklearn.preprocessing import LabelEncoder
#enc = LabelEncoder()
#Y = np.reshape(Y,(250000))
#enc.fit(Y)
#Y = enc.transform(Y)

Y.replace(["s","b"], [1,0], inplace=True)


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)




