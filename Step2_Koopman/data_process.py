# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 21:10:51 2022

@author: chongloc
"""
import pandas as pd
import numpy as np
#import netCDF4 as nc
#data=nc.Dataset('./data/State data/2019/zurich_2019_week10.nc')
#print(data.variables.keys())
#data.variables['R1'][55,:].compressed().shape

def Get_shape_filter(data):
    #data:n*33*212*34
    D,M,N=data.shape
    select_data=[]
    loc_log={}
    k=0
    shape_filter=np.ones((D,M,N))
    for d in range(D):
        for m in range(M):
            for n in range(N):
                if data[d,m,n]!=-999:
                    shape_filter[d,m,n]=0
                    select_data.append(data[d,m,n])
                    loc_log[k]=(d,m,n)
                    k+=1
    return shape_filter,np.array(select_data),loc_log

def Reshape_data(x,loc_log,original_shape):
    #use loc_log to reconstructure the original data with shape 33*212*34
    X=np.full(original_shape,np.nan)
    n=x.shape[0]
    for i in range(n):
        d,m,n=loc_log[i]
        X[d,m,n]=x[i]
    return X

#shape_filter,select_data,loc_log=Get_shape_filter(data.variables['R1'][0,:])
#X=Reshape_data(select_data,loc_log,(33,212,34))

'''
def get_data(year,selected):
    DATA_U,DATA_V,DATA_R=[],[],[]
    for week in selected:
        data=nc.Dataset('./data/State data/'+year+'/zurich_'+year+'_week'+str(week)+'.nc')
        U=data.variables['U1'][:]
        V=data.variables['V1'][:]
        R=data.variables['R1'][:]

        for t in range(U.shape[0]):
            U_shape_filter,U_data,U_loc_log=Get_shape_filter(U[t])
            DATA_U.append(U_data.tolist())
            V_shape_filter,V_data,V_loc_log=Get_shape_filter(V[t])
            DATA_V.append(V_data.tolist())
            R_shape_filter,R_data,R_loc_log=Get_shape_filter(R[t])
            DATA_R.append(R_data.tolist())

    DATA_U=np.array(DATA_U)
    DATA_V=np.array(DATA_V)
    DATA_R=np.array(DATA_R)
    print(DATA_U.shape,DATA_V.shape,DATA_R.shape)
    
    DATA_F=[]
    for week in selected:
        U=pd.read_excel('./data/Forcing data/'+year+'/zurich_'+year+'_week'+str(week)+'.xlsx').values[:,3:]

        for t in range(U.shape[0]):
            DATA_F.append(U[0].tolist())

    DATA_F=np.array(DATA_F)
    print(DATA_F.shape)
    return DATA_U,DATA_V,DATA_R,DATA_F,U_loc_log,V_loc_log,R_loc_log    
'''

'''
Normaized data and select useful data
'''

def normlize_F(data):
    n,m=data.shape
    tdata=np.zeros(data.shape)
    dm=data.copy()
    dmax=np.max(data,axis=0)
    dmin=np.min(data,axis=0)
    for i in range(n):
        for j in range(m):
            if dmax[j]==dmin[j]:
                tdata[i,j]=0
            else:
                tdata[i,j]=2*((dm[i,j]-dmin[j])/(dmax[j]-dmin[j]))-1
    return tdata,dmax,dmin

def renormalize_F(tdata,dmax,dmin):
    n,m=tdata.shape
    data=np.zeros(tdata.shape)
    for i in range(n):
        for j in range(m):
            if dmax[j]==dmin[j]:
                data[i,j]=dmax[j]
            else:
                data[i,j]=0.5*(tdata[i,j]+1)*(dmax[j]-dmin[j])+dmin[j]
    return data

def normlize(data):
    n,m=data.shape
    tdata=np.zeros(data.shape)
    dm=data.copy()
    dmax=np.max(data,axis=0)
    dmin=np.min(data,axis=0)
    for i in range(n):
        tdata[i]=2*((dm[i]-dmin)/(dmax-dmin))-1
    return tdata,dmax,dmin

def renormalize(tdata,dmax,dmin):
    n,m=tdata.shape
    data=np.zeros(tdata.shape)
    for i in range(n):
        data[i]=0.5*(tdata[i]+1)*(dmax-dmin)+dmin
    return data

def trans_data_process(DATA_U,DATA_V,DATA_R,DATA_F):
    t_DATA_F,fmax,fmin=normlize_F(DATA_F)
    t_DATA_U,umax,umin=normlize(DATA_U)
    t_DATA_V,vmax,vmin=normlize(DATA_V)
    t_DATA_R,rmax,rmin=normlize(DATA_R)
    
    t_DATA_F=DATA_F
    t_DATA_U=DATA_U
    t_DATA_V=DATA_V
    t_DATA_R=DATA_R

    return t_DATA_F,t_DATA_U,t_DATA_V,t_DATA_R


def training_data_process(DATA_U,DATA_V,DATA_R,DATA_F,selected):
    
    t_DATA_F,fmax,fmin=normlize_F(DATA_F)
    t_DATA_U,umax,umin=normlize(DATA_U)
    t_DATA_V,vmax,vmin=normlize(DATA_V)
    t_DATA_R,rmax,rmin=normlize(DATA_R)
    
    t_DATA_F=DATA_F
    t_DATA_U=DATA_U
    t_DATA_V=DATA_V
    t_DATA_R=DATA_R
    
    Utrain=[]
    Vtrain=[]
    Rtrain=[]
    Ftrain=[]
    Utrain_out=[]
    Vtrain_out=[]
    Rtrain_out=[]

    for t in selected:
        Utrain.append(t_DATA_U[t,:].tolist())
        Vtrain.append(t_DATA_V[t,:].tolist())
        Rtrain.append(t_DATA_R[t,:].tolist())
        Ftrain.append(t_DATA_F[t,:].tolist())

        Utrain_out.append(t_DATA_U[t+1,:].tolist())
        Vtrain_out.append(t_DATA_V[t+1,:].tolist())
        Rtrain_out.append(t_DATA_R[t+1,:].tolist())

    Utrain=np.array(Utrain)
    Vtrain=np.array(Vtrain)
    Rtrain=np.array(Rtrain)
    Ftrain=np.array(Ftrain)
    Utrain_out=np.array(Utrain_out)
    Vtrain_out=np.array(Vtrain_out)
    Rtrain_out=np.array(Rtrain_out)
    return Utrain,Vtrain,Rtrain,Ftrain,Utrain_out,Vtrain_out,Rtrain_out
