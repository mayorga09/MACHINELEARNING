# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 15:37:52 2025

@author: jramiro
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 12:09:07 2025

@author: jramiro
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import statsmodels.api as sm



def PulgtoCM (a):
    Pulg=a[a['Ancho_Largo_UOM_c']=='PULG']
    CMS=a[a['Ancho_Largo_UOM_c']=='CMS']
    Pulg['Ancho_CMS']=Pulg['Ancho_c']*2.54
    CMS['Ancho_CMS']=CMS['Ancho_c']*1
    base=pd.concat([Pulg,CMS])
    return base



Clusters=pd.read_excel("C:/Users/jramiro/OneDrive - PRODUCTORA DE PAPEL, S.A. DE C.V/Documentos/Scripts Py/BaseFINAL2.xlsx")
TD_Parte=pd.read_excel("Catalogo Parte.xlsx")
Parte=pd.read_excel("Partes_Limpias.xlsx")
IGAE=pd.read_excel('IGAE.xlsx')
Fact2018=pd.read_csv('Fact 2018.csv')

Fact2018=Fact2018[Fact2018['Ancho_Largo_UOM_c'].isin(['CMS','PULG'])]

Fact2025=pd.read_csv("C:/Users/jramiro/OneDrive - PRODUCTORA DE PAPEL, S.A. DE C.V/Documentos/Scripts Py/Fact 2024-2025.csv")
Fact2023=pd.read_csv("Fact 2022-2023.csv")
Base=pd.concat([Fact2025,Fact2023])
Base=Base.dropna()
Base=PulgtoCM(Base)

Base=pd.merge(Base, Clusters[['Name','Cluster_gmm']], on='Name')
Base=pd.merge(Base, TD_Parte, left_on='PartNum',right_on='Parte')
Base=pd.merge(Base,Parte,left_on='PartNum',right_on='Partes')

Fact2018=pd.merge(Fact2018, Clusters[['Name','Cluster_gmm']], on='Name')
Fact2018=pd.merge(Fact2018, TD_Parte, left_on='PartNum',right_on='Parte')
Fact2018=pd.merge(Fact2018,Parte,left_on='PartNum',right_on='Partes')


IGAE['Periodo']=pd.to_datetime(IGAE['Periodo'],format='%Y/%m')
IGAE['Mes']=IGAE['Periodo'].dt.to_period('M')
IGAE['CMes']=IGAE['Periodo'].dt.month

Base['InvoiceDate']=pd.to_datetime(Base['InvoiceDate'])
Base['Mes']=Base['InvoiceDate'].dt.to_period('M')
Base['CMes']=Base['InvoiceDate'].dt.month
Base['Año']=Base['InvoiceDate'].dt.year
Base['Fecha_Ly']=Base['InvoiceDate']-pd.DateOffset(years=1)
Base['MesLY']=Base['Fecha_Ly'].dt.to_period('M')
Fact2018['InvoiceDate']=pd.to_datetime(Fact2018['InvoiceDate'])
Fact2018['Mes']=Base['InvoiceDate'].dt.to_period('M')
Fact2018['CMes']=Base['InvoiceDate'].dt.month
Fact2018['Año']=Base['InvoiceDate'].dt.year



### Agrupado Por Cluster y Familia
#%%
agg_df=Base.groupby(['Mes','Fam','Cluster_gmm','CMes','MesLY'],
                    as_index=False).agg({'TON':'sum','FactNeto':'sum'})

agg_df['Precio']=agg_df['FactNeto']/agg_df['TON']

agg_df=agg_df[(agg_df['TON']>0) & (agg_df['FactNeto']>0)]


agg_18=Fact2018.groupby(['CMes','Fam','Cluster_gmm'],
                    as_index=False).agg({'TON':'sum','FactNeto':'sum'})

agg_18['Precio']=agg_18['FactNeto']/agg_df['TON']

agg_18=agg_18[(agg_18['TON']>0) & (agg_18['FactNeto']>0)]

agg_18=agg_18.rename(columns={'TON':'2018'})



agg_df=pd.merge(agg_df,IGAE[['Indice','Mes']], on='Mes')

agg_df=pd.merge(agg_df, agg_18[['Fam','Cluster_gmm','CMes','2018']], on=['Fam','Cluster_gmm','CMes'])

agg_df['Est_IGAE']=(agg_df['Indice']/100)*agg_df['2018']

agg_df['Indice_Demanda']=agg_df['TON']/agg_df['Est_IGAE']


agg_df_ly=agg_df[['Cluster_gmm','Fam','Mes','Precio','Indice_Demanda']]

agg_df_ly=agg_df_ly.rename(columns={'Mes':'MesLY','Precio':'Precio_LY','Indice_Demanda':'Indice_Demanda_LY'})

agg_df=pd.merge(agg_df,agg_df_ly, on=['Cluster_gmm','Fam','MesLY'])


agg_df['ln_Precio'] = np.log(agg_df['Precio'])
agg_df['ln_Demanda'] = np.log(agg_df['Indice_Demanda'])

agg_df['ln_PrecioLY'] = np.log(agg_df['Precio_LY'])
agg_df['ln_DemandaLY'] = np.log(agg_df['Indice_Demanda_LY'])

results = []
for part, data in agg_df.groupby('Fam'):
    if data['ln_Precio'].nunique() > 1:
        X = sm.add_constant(data['ln_Precio'])
        y = data['ln_Demanda']
        model = sm.OLS(y, X).fit()
        results.append({
            'PartNum': part,
            'Elasticidad_Ajustada': model.params['ln_Precio'],
            'R2': model.rsquared,
            'p_value': model.pvalues['ln_Precio']
        })
        
elasticidad_ajustada_df = pd.DataFrame(results)

agg_df['dln_Precio']=agg_df['ln_Precio']-agg_df['ln_PrecioLY']
agg_df['dln_Demanda']=agg_df['ln_Demanda']-agg_df['ln_DemandaLY']


results = []

for (fam,cluster), data in agg_df.dropna(subset=['dln_Precio', 'dln_Demanda']).groupby(['Fam','Cluster_gmm']):
    if data['dln_Precio'].nunique() > 1:
        X = sm.add_constant(data['dln_Precio'])
        y = data['dln_Demanda']
        model = sm.OLS(y, X).fit()
        results.append({
            'Fam': fam,
            'Cluster':cluster,
            'Elasticidad': model.params['dln_Precio'],
            'R2': model.rsquared,
            'p_value': model.pvalues['dln_Precio'],
            'n_obs': len(data)
        })

elasticidad_mes_a_mes = pd.DataFrame(results)
elasticidad_mes_a_mes = elasticidad_mes_a_mes.sort_values('Elasticidad')


#%% #Agrupado por Cluster
agg_df=Base.groupby(['Mes','Cluster_gmm','CMes','MesLY'],
                    as_index=False).agg({'TON':'sum','FactNeto':'sum'})

agg_df['Precio']=agg_df['FactNeto']/agg_df['TON']

agg_df=agg_df[(agg_df['TON']>0) & (agg_df['FactNeto']>0)]


agg_18=Fact2018.groupby(['CMes','Cluster_gmm'],
                    as_index=False).agg({'TON':'sum','FactNeto':'sum'})

agg_18['Precio']=agg_18['FactNeto']/agg_df['TON']

agg_18=agg_18[(agg_18['TON']>0) & (agg_18['FactNeto']>0)]

agg_18=agg_18.rename(columns={'TON':'2018'})



agg_df=pd.merge(agg_df,IGAE[['Indice','Mes']], on='Mes')

agg_df=pd.merge(agg_df, agg_18[['Cluster_gmm','CMes','2018']], on=['Cluster_gmm','CMes'])

agg_df['Est_IGAE']=(agg_df['Indice']/100)*agg_df['2018']

agg_df['Indice_Demanda']=agg_df['TON']/agg_df['Est_IGAE']


agg_df_ly=agg_df[['Cluster_gmm','Mes','Precio','Indice_Demanda']]

agg_df_ly=agg_df_ly.rename(columns={'Mes':'MesLY','Precio':'Precio_LY','Indice_Demanda':'Indice_Demanda_LY'})

agg_df=pd.merge(agg_df,agg_df_ly, on=['Cluster_gmm','MesLY'])


agg_df['ln_Precio'] = np.log(agg_df['Precio'])
agg_df['ln_Demanda'] = np.log(agg_df['Indice_Demanda'])

agg_df['ln_PrecioLY'] = np.log(agg_df['Precio_LY'])
agg_df['ln_DemandaLY'] = np.log(agg_df['Indice_Demanda_LY'])

results = []
for cluster, data in agg_df.groupby('Cluster_gmm'):
    if data['ln_Precio'].nunique() > 1:
        X = sm.add_constant(data['ln_Precio'])
        y = data['ln_Demanda']
        model = sm.OLS(y, X).fit()
        results.append({
            'Cluster': cluster,
            'Elasticidad_Ajustada': model.params['ln_Precio'],
            'R2': model.rsquared,
            'p_value': model.pvalues['ln_Precio']
        })
        
elasticidad_ajustada_df_cluster = pd.DataFrame(results)

agg_df['dln_Precio']=agg_df['ln_Precio']-agg_df['ln_PrecioLY']
agg_df['dln_Demanda']=agg_df['ln_Demanda']-agg_df['ln_DemandaLY']


results = []

for cluster, data in agg_df.dropna(subset=['dln_Precio', 'dln_Demanda']).groupby('Cluster_gmm'):
    if data['dln_Precio'].nunique() > 1:
        X = sm.add_constant(data['dln_Precio'])
        y = data['dln_Demanda']
        model = sm.OLS(y, X).fit()
        results.append({
            'Cluster':cluster,
            'Elasticidad': model.params['dln_Precio'],
            'R2': model.rsquared,
            'p_value': model.pvalues['dln_Precio'],
            'n_obs': len(data)
        })

elasticidad_mes_a_mes_cluster = pd.DataFrame(results)
elasticidad_mes_a_mes_cluster = elasticidad_mes_a_mes_cluster.sort_values('Elasticidad')
#%% #Agrupado por Familia
