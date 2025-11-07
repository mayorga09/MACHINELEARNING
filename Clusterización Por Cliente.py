import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.neighbors import NearestNeighbors

def PulgtoCM (a):
    Pulg=a[a['Ancho_Largo_UOM_c']=='PULG']
    CMS=a[a['Ancho_Largo_UOM_c']=='CMS']
    Pulg['Ancho_CMS']=Pulg['Ancho_c']*2.54
    CMS['Ancho_CMS']=CMS['Ancho_c']*1
    base=pd.concat([Pulg,CMS])
    return base

def cluster_metrics(X, labels, name):
    mask = labels != -1  
    if len(set(labels[mask])) < 2:
        return -1
    
    sil = silhouette_score(X[mask], labels[mask])
    ch = calinski_harabasz_score(X[mask], labels[mask])
    db = davies_bouldin_score(X[mask], labels[mask])
    
    return sil,ch,db

def plot_3d(X, labels, title):
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], c=labels, s=40)
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()

def plot_k_distance(X, k=5):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    d_sorted = np.sort(distances[:, -1])
    return d_sorted
    
    """plt.figure(figsize=(6,3))
    plt.plot(d_sorted)
    plt.title(f"K-distance Plot (k={k}) — busca codo")
    plt.xlabel("Puntos ordenados")
    plt.ylabel(f"Distancia al {k}° vecino más cercano")
    plt.grid(True)
    plt.show()"""

Base=pd.read_csv("C:/Users/jramiro/OneDrive - PRODUCTORA DE PAPEL, S.A. DE C.V/Documentos/Scripts Py/Fact 2024-2025.csv")
Base=Base.dropna()
Base=PulgtoCM(Base)

Base['InvoiceDate']=pd.to_datetime(Base['InvoiceDate'])
Base['Trimestre']=Base['InvoiceDate'].dt.to_period("Q")
Base=Base[Base['InvoiceDate'].dt.year==2024]
Base=Base[Base['Trimestre']!='2024Q4']

Agrupado= Base.groupby(['Name','Trimestre']).agg({'TON':'sum'
                                                                ,'FactNeto':'sum'
                                                                ,'Ancho_CMS':[np.std,np.mean]
                                                                ,'PartNum':'nunique'}).reset_index()
Agrupado.columns = ['_'.join(col).strip('_') for col in Agrupado.columns.values]
Agrupado=Agrupado.dropna()
Agrupado['Precio']=Agrupado['FactNeto_sum']/Agrupado['TON_sum']
Agrupado=Agrupado.dropna()

RFM={}
Metrics_HDBSCan=pd.DataFrame()
Metrics_Optics=pd.DataFrame()
Metrics_DBSCan=pd.DataFrame()
Metrics_Gaussian=pd.DataFrame()
aux=pd.DataFrame()
aux2=pd.DataFrame()
for i in Agrupado['Trimestre'].unique():
    aux=Agrupado[Agrupado['Trimestre']==i]
    Q1 = aux['TON_sum'].quantile(0.25)
    Q3 = aux['TON_sum'].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    aux=aux[(aux['TON_sum']>=lim_inf) & (aux['TON_sum']<=lim_sup) ]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(aux[['TON_sum',
           'Ancho_CMS_std', 'Ancho_CMS_mean', 'Precio','PartNum_nunique']])
    X_Scaled=scaled_data
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(X_Scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    print(explained_variance_ratio.sum())
    
    PCA_Df=pd.DataFrame(principal_components)

    PCA_Df=PCA_Df.rename(columns={0:'PC1',1:'PC2',2:'PC3'})

    aux=pd.concat([aux.reset_index(),PCA_Df],axis=1)

    aux.plot.scatter(x='PC1', y='PC2')
    plt.show() 

    X=aux[aux['Trimestre']==i][['PC1','PC2','PC3']]
    for j in range(2,30):
        hdb = hdbscan.HDBSCAN(min_cluster_size=j).fit(X)
        labels_hdb = hdb.labels_
        if cluster_metrics(X, labels_hdb, "HDBSCAN") == -1:
            print(cluster_metrics(X, labels_hdb, "HDBSCAN"))
            continue
        a,b,c=cluster_metrics(X, labels_hdb, "HDBSCAN") 
        aux2['Trimestre']=[i]
        aux2['Clusters']=[j]
        aux2['Sil']=[a]
        aux2['CH']=[b]
        aux2['DB']=[c]
        Metrics_HDBSCan=pd.concat([Metrics_HDBSCan,aux2])

    for j in range(2,30):
       db = DBSCAN(eps=0.8, min_samples=j).fit(X)
       labels_db = db.labels_
       if cluster_metrics(X, labels_db, "DBSCAN") == -1:
           continue
       a,b,c=cluster_metrics(X, labels_db, "DBSCAN (eps=0.8)")
       aux2['Trimestre']=i
       aux2['Clusters']=j
       aux2['Sil']=a
       aux2['CH']=b
       aux2['DB']=c
       Metrics_DBSCan=pd.concat([Metrics_DBSCan,aux2])
       
    for j in range(2,30):
       op = OPTICS(min_samples=j).fit(X)
       labels_opt = op.labels_
       if cluster_metrics(X, labels_opt, "OPTICS") == -1:
           continue
       a,b,c=cluster_metrics(X, labels_opt, "OPTICS")
       aux2['Trimestre']=i
       aux2['Clusters']=j
       aux2['Sil']=a
       aux2['CH']=b
       aux2['DB']=c
       Metrics_Optics=pd.concat([Metrics_Optics,aux2])
   
    for j in range(2,30):
      gmm = GaussianMixture(n_components=j, random_state=42).fit(X)
      labels_gmm = gmm.predict(X)
      if cluster_metrics(X, labels_gmm, "GMM") == -1:
          continue
      a,b,c=cluster_metrics(X, labels_gmm, "GMM")
      aux2['Trimestre']=i
      aux2['Clusters']=j
      aux2['Sil']=a
      aux2['CH']=b
      aux2['DB']=c
      Metrics_Gaussian=pd.concat([Metrics_Gaussian,aux2])
      
      
      
# === LLAMAR PARA VER VALOR DE EPS ===
plot_k_distance(X, k=5)

#DBSCAN K=4 GAUSSIAN K=4 HDSCAN K=4 OPTICS K=4

for i in Agrupado['Trimestre'].unique():
    aux=Agrupado[Agrupado['Trimestre']==i]
    Q1 = aux['TON_sum'].quantile(0.25)
    Q3 = aux['TON_sum'].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    aux=aux[(aux['TON_sum']>=lim_inf) & (aux['TON_sum']<=lim_sup) ]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(aux[['TON_sum',
           'Ancho_CMS_std', 'Ancho_CMS_mean', 'Precio','PartNum_nunique']])
    X_Scaled=scaled_data
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(X_Scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    print(explained_variance_ratio.sum())
    
    PCA_Df=pd.DataFrame(principal_components)

    PCA_Df=PCA_Df.rename(columns={0:'PC1',1:'PC2',2:'PC3'})

    aux=pd.concat([aux.reset_index(),PCA_Df],axis=1)

    aux.plot.scatter(x='PC1', y='PC2')
    plt.show() 

    X=aux[aux['Trimestre']==i][['PC1','PC2','PC3']]
    hdb = hdbscan.HDBSCAN(min_cluster_size=4).fit(X)
    labels_hdb = hdb.labels_
    db = DBSCAN(eps=0.8, min_samples=4).fit(X)
    labels_db = db.labels_
    op = OPTICS(min_samples=4).fit(X)
    labels_opt = op.labels_
    gmm = GaussianMixture(n_components=4, random_state=42).fit(X)
    labels_gmm = gmm.predict(X)
    aux['Cluster_HDB']=labels_hdb
    aux['Cluster_DB']=labels_db
    aux['Cluster_op']=labels_opt
    aux['Cluster_gmm']=labels_gmm
    RFM.update({i:aux})
    

BaseFinal=pd.DataFrame()
for i in RFM:
    BaseFinal=pd.concat([BaseFinal,RFM[i]])
    
BaseFinal.to_excel('BaseFINAL.xlsx')

        
"""pivot=pd.pivot_table(CH_index,values='CH_index',index='Clusters',columns='Trimestre',aggfunc='mean')

fig, ax = plt.subplots()

ax.plot(pivot[['2024Q1']], label='Q1', marker='v')
ax.plot(pivot[['2024Q2']], label='Q2', marker='v')
ax.plot(pivot[['2024Q3']], label='Q3', marker='v')

ax.set_xlabel('Clusters')
ax.set_ylabel('index')
ax.set_title('codo')

ax.legend()

plt.show() 

##Se determinaron 4 Clusters

for i in df['Trimestre'].unique():
    aux=df[df['Trimestre']==i]
    X=df[df['Trimestre']==i][['PC1','PC2']]
    kmeans=KMeans(n_clusters=5,random_state=0).fit(X)
    aux['Kmeans']=kmeans.labels_
    RFM.update({i:aux}) """
    

    
    


