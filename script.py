# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import sqrt

# %%
class script:
    def __init__(self):
       self.tags=[]
       self.number=None
       self.method=None
       self.fdf=None
       self.weightslist=None
       self.inpw=[]
       self.cm=None
       self.means=None
       self.switch=0
       self.minriskweights=None
       self.market_cap_weights=None
       self.shrpqweights=None
       self.shrpweights=None
    def gentags(self,tglst):
        self.tags=tglst.split()
        return self.tags
    
    def inpweights(self,w):
        self.switch=1
        w=w.split()
        print(w,type(w))
        w=list(map(float,w))
        w=np.array(w)
        w=np.array(w)/np.array(w).sum()
        print(w)
        stdf=sqrt(w.dot(self.cm).dot(w.T))
        retf=w.dot(self.means)
        self.inpw=[stdf,retf]
        return None
    
    def scriptm(self):
        # tags= ["MSFT","AAPL","HMY","WMT","MC","ADM","KO","NSRGY","F","GE"]
        # tag_MC = ["UBS","NYCB","FRC","NIO","TATAMOTORS.NS"]
        # tags1=tags+tag_MC
        print(self.tags)
        len(self.tags) #tags of required companies are taken as a list, this is to be generalised number of stocks and the names.

        # %%
        #####!pip install yfinance

        # %%
        dic={}
        tags=self.tags
        import yfinance as yf
        minr=yf.Ticker(tags[0]).history(period="10y").shape[0]
        for i in range(len(tags)):
            if yf.Ticker(tags[i]).history(period="10y").shape[0]<=minr:
                minr=yf.Ticker(tags[i]).history(period="10y").shape[0]
        for i in range(len(tags)):
            dic[f"{i}"]=yf.Ticker(tags[i]).history(period="10y").iloc[:minr,:] 
            dic[f'{i}']['Closeprev']=np.append([np.array(0)],(dic[f'{i}'].Close.iloc[:-1]))
        #dictionary is created taking i as the key and the entire data frame returened by yf library as the value.
        #minimum of rows of all stocks is to be taken instead of 250

        # %%
        # for i in range(len(tags)):
        # #   print(dic[f'{i}'].shape[0]) #just to check if all the values of dictionary have same number of values
        # print(np.append([np.array(0)],(dic[f'{i}'].Close.iloc[:-1])))
        # ver= pd.DataFrame(dic['8'])
        # ver
        # df['1'].shape

        # %%
        rdic={}
        for i in range(len(tags)):
            rdic[f'{i}']=(np.array(dic[f'{i}'].Close.iloc[1:])-np.array(dic[f'{i}'].Closeprev.iloc[1:]))/(np.array(dic[f'{i}'].Closeprev.iloc[1:])*0.01)
            #a dictionary to calculate returns for the given companies is calculated

        # %%
        # for i in range(len(tags)):
        #   print(len(rdic[f'{i}']))
        #  #just to check if all the values of dictionary have same number of values
        rdic={}
        for i in range(len(tags)):
            rdic[f'{i}']=(np.array(dic[f'{i}'].Close.iloc[1:])-np.array(dic[f'{i}'].Closeprev.iloc[1:]))/(np.array(dic[f'{i}'].Closeprev.iloc[1:])*0.01)
            #a dictionary to calculate returns for the given companies is calculated

        # %%
        df=pd.DataFrame(rdic)
        df

        # %%
        cm=df.cov()
        cm

        # %%
        cm.shape

        # %%
        mdf=df.mean()
        means=np.array(mdf)
        means=means.reshape(mdf.shape[0],1)
        means.shape

        # %%
        # df.iloc[:,1].sum()

        # %%


        # %%
        if self.method=='Monte_Carlo':
            tsd=[]
            tm=[]
            weightslist=[]
            for i in range(int(self.number)):
                w=np.random.random(size=(len(tags),1)) #no of weights is to be generalised
                w=w/(w.sum())
                weightslist.append(w)
                tsd.append(sqrt(np.ravel(w.T.dot(cm).dot(w))))
                tm.append(w.T.dot(means))
        elif self.method=='Quasi_MC':
            tsd=[]
            tm=[]
            weightslist=[]
            from scipy.stats import qmc
            dist = qmc.MultivariateNormalQMC(mean=means.T.ravel(), cov=cm.to_numpy())
            sample = dist.random(int(self.number))
            sample=abs(sample)
            sample
            for i in range(len(sample)):
                sample[i]=sample[i]/sample[i].sum()
            sample=sample.tolist()
            weightslist=sample
            for i in weightslist:
                i=np.array(i)
                tsd.append(sqrt(np.ravel(i.dot(cm).dot(i.T))))
                tm.append(i.dot(means))
        
        elif self.method=='Sobol':
            tsd=[]
            tm=[]
            weightslist=[]
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=len(tags), scramble=False)
            sample = sampler.random_base2(m=int(math.floor(((math.log(int(self.number))/math.log(2))+1))))
            sample = sample[:int(self.number)]
            sample=abs(sample)
            sample
            for i in range(len(sample)):
                sample[i]=sample[i]/sample[i].sum()
            sample=sample.tolist()
            weightslist=sample
            for i in weightslist:
                i=np.array(i)
                tsd.append(sqrt(np.ravel(i.dot(cm).dot(i.T))))
                tm.append(i.dot(means))

        else:
            print('@@@@@@@@@@@@@@Give some method@@@@@@@@@@@@@@')
        # #######plt.scatter(tsd,tm,edgecolor='black',alpha=0.7)
        # #######plt.xlabel('standard deviation')
        # #######plt.ylabel('mean returns')
        #applying monte carlo method of randomly selecting array of weights from a uniform distribution ans making their sum equal to one
        #we have calculated portfolio's standard deviatino and mean and plotted it down

        # %%
        tsd=np.array(tsd)
        tsd=np.ravel(tsd)
        tm=np.array(tm)
        tm=np.ravel(tm)

        # %%
        fdic={}
        fdic["means"]=tm.tolist()
        fdic["vars"]=tsd.tolist()
        fdf=pd.DataFrame(fdic)
        tsdl=np.sort(tsd).tolist()

        # %%
        l=0.03
        maax=[]
        ptsd=[]
        nparr= np.arange(tsdl[1],tsdl[-2],l)
        for i in range(len(nparr.tolist())):
            # print(i)
            if fdf[(fdf["vars"]>nparr[i]-l) & (fdf["vars"]<nparr[i]+l)].means.shape==(0,):
                continue
            gh=max(fdf[(fdf["vars"]>nparr[i]-l) & (fdf["vars"]<nparr[i]+l)].means)
            maax.append(gh)
            ptsd.append(max(fdf[((fdf["vars"]>nparr[i]-l) & (fdf["vars"]<nparr[i]+l)) & (fdf["means"]==gh)].vars))
        print(len(maax),len(ptsd))

        # %%
        # import math
        # #######plt.scatter(tsd,tm,edgecolor='black',alpha=0.7)
        # #######plt.scatter(ptsd[:-math.floor(0.10*len(ptsd))],maax[:-math.floor(0.1*len(ptsd))],color='red')
        # #######plt.scatter(frontierstd,frontierret,color='green',alpha=0.3)

        # %%
        #####!pip install scipy

        # %%
        # from scipy.interpolate import make_interp_spline #cell doesn't run because we have duplicate values of max return for a range of variance.
        # g=math.floor(0.020*len(ptsd))
        # h=math.floor(0.02*len(maax))
        # ptsd_maax=make_interp_spline(ptsd[:-g],maax[:-h])
        # print(ptsd_maax)
        # ptsd_=np.linspace(ptsd[0],ptsd[-1], 25)
        # maax_=ptsd_maax(ptsd_)
        # plt.plot(ptsd_,maax_,color='red')
        # import math
        # plt.scatter(tsd,tm,edgecolor='black',alpha=0.7)
        # # plt.plot(ptsd[:-math.floor(0.020*len(ptsd))],maax[:-math.floor(0.02*len(ptsd))],color='red')

        # %%
        # pd.DataFrame(np.array(ptsd)).value_counts()

        # %%
        #####!pip install sklearn
        from sklearn.linear_model import LinearRegression #fitting a line using polynomial regression to fit the line going through portfolios of efficient frontier 
        from sklearn.preprocessing import PolynomialFeatures
        model=LinearRegression()
        dfx=pd.DataFrame(np.array(ptsd))
        dfy=pd.DataFrame(np.array(maax))
        poly=PolynomialFeatures(degree=10,include_bias=True).fit_transform(dfx)
        dfx=pd.DataFrame(poly)
        model.fit(dfx,dfy)


        # %%
        treasury=yf.Ticker('^TNX').history(period="10y")
        treasury['Closeprev']=np.append([np.array(0)],(treasury.Close.iloc[:-1]))
        treasuryret=pd.DataFrame((np.array(treasury.Close.iloc[1:])-np.array(treasury.Closeprev.iloc[1:]))/(np.array(treasury.Closeprev.iloc[1:])*0.01))
        tmdf=treasuryret.mean(axis=0)
        print(f'Risk free return is {tmdf.iloc[0]}')
        rfr=tmdf.iloc[0]
        treasuryret #risk free rate of return is calculated using 10 year treasury bond opening and closing rates

        # %%
        shrp=(tm[0]-rfr)/tsd[0]
        shrpindx=0
        for i in range(len(tsd)): #sharpe point at which the ratio of return premium over risk free return to risk is the highest
            if (tm[i]-rfr)/tsd[i] >= shrp: #
                shrp=(tm[i]-rfr)/tsd[i]
                shrpindx=i
        minsd=fdf[fdf["vars"]==fdf.vars.min()].vars
        index=fdf[fdf["vars"]==fdf.vars.min()].index.tolist()
        print(f'max sharpe ratio is {shrp}, occuring at {shrpindx} point in the sorted list of risks, \n thus portfolio having sd as {tsd[shrpindx]} and returns as {tm[shrpindx]} is the most optimum')
        print(f'The weights given to each stock being \n {weightslist[shrpindx]}')
        self.shrpweights=weightslist[shrpindx]
        # print(tm[index[0]])
        print(f'Min risk point is {minsd.tolist()[0]},{tm[index[0]]} with weights as {weightslist[index[0]]}')
        self.minriskweights=weightslist[index[0]]

        # %%
        ypred=model.predict(dfx)
        # ####### plt.plot(ptsd,ypred.tolist(),color='red')
        # #######plt.scatter(tsd,tm,alpha=0.7)
        # #######plt.scatter(tsd[shrpindx],tm[shrpindx],color='red',edgecolor='black',label='Sharpe point',alpha=0.7)
        # #######plt.scatter(minsd.tolist()[0],tm[index[0]],color='yellow',edgecolor='black',label='Min variance point',alpha=0.7)
        # #######plt.legend()
        # plt.scatter(frontierstd,frontierret,color='green',alpha=0.3)

        

        # %%
        #####!pip install gurobipy #finding efficient frontier using quadratic programming

        # %%
        #covariance matrix si Q and w matix is the weights which are to be optimised
        Q=np.array(cm)
        leg=len(tags)
        #w is the array of variables to be optimized
        #constraints are summation is 1 and nothing is said about negativity
        A=np.ones((leg))
        #constraint would be ax=b wehre x is w (a colomn vector of size leg) and b is 1



        # %%
        A.shape

        # %%
        import gurobipy as gp
        m=gp.Model('portfolio')
        x=m.addMVar(len(tags))
        portfolio_risk=x @ Q @ x
        m.setObjective(portfolio_risk)
        m.addConstr(x.sum()==1)
        m.optimize()
        print(sqrt(m.ObjVal))
        varnames=m.getVars()
        values = m.getAttr("X",varnames)
        print(values)

        # %%
        #for diffeent risks add the constraint to model that the risk is equal to the given risk 
        #and then make a list of different sd's at different risk
        looplst=np.arange(mdf.min(),mdf.max(),0.003)
        mdfa=np.array(mdf)
        target=m.addConstr(mdfa @ x == 0.07, 'target')
        frontierstd=[]
        frontierret=[]
        frontierweights=[]
        for i in range(len(looplst)):
            target.rhs=looplst[i]
            m.optimize()
            frontierstd.append(sqrt(m.ObjVal))
            frontierret.append(looplst[i])
            varnames=m.getVars()
            values = m.getAttr("X",varnames)
            frontierweights.append(values)
        plt.scatter(frontierstd,frontierret)

        # %%
        shrpq=(frontierret[0]-0.0099)/frontierstd[0]
        shrpqindx=0
        for i in range(len(frontierstd)):
            if (frontierret[i]-0.0099)/frontierstd[i] >= shrpq:
                shrpq=(frontierret[i]-0.0099)/frontierstd[i]
                shrpqindx=i
        self.shrpqweights=frontierweights[shrpqindx]
        print(f"Sharpe point according to quad optimization is at weights {frontierweights[shrpqindx]}")
        #%%
        # plt.plot(ptsd,ypred.tolist(),color='red')
        # plt.scatter(tsd,tm,alpha=0.7)
        # plt.scatter(tsd[shrpindx],tm[shrpindx],color='red',edgecolor='black',label='Sharpe point',alpha=0.7)
        # plt.legend()
        # plt.scatter(frontierstd,frontierret)

        # %%
        from pandas_datareader import data as web
        market_cap_data = web.get_quote_yahoo(tags)['marketCap']
        market_cap_data=pd.DataFrame(market_cap_data)
        market_cap_weights=(np.array(market_cap_data.marketCap)/np.array(market_cap_data.marketCap).sum())
        self.market_cap_weights=market_cap_weights
        market_cap_sd=sqrt(market_cap_weights.dot(cm).dot(market_cap_weights.T))
        market_cap_mean=mdf.dot(market_cap_weights)
        # weithgts of portfolio taken based on market capitalisation of each stock
        # %%
        fig, ax = plt.subplots(1,1,figsize=(6,4))
        ax.scatter(tsd,tm,alpha=0.7)
        ax.scatter(ptsd,maax,color='black',label='Effecient frontier',edgecolor='black',alpha=0.7)
        ax.scatter(frontierstd,frontierret,label='Quadratic optimization')
        ax.scatter(tsd[shrpindx],tm[shrpindx],color='red',edgecolor='black',label='Sharpe point',alpha=1.0)
        ax.scatter(frontierstd[shrpqindx],frontierret[shrpqindx],color='purple', edgecolor='black',label='Quadratic Sharpe point',alpha=0.7)
        ax.scatter(market_cap_sd,market_cap_mean,color='green',edgecolor='black',label='Market Cap weights',alpha=1.0)
        ax.scatter(minsd.tolist()[0],tm[index[0]],color='yellow',edgecolor='black',label='Min variance point',alpha=1.0)
        if self.switch==1:
            ax.scatter(self.inpw[0],self.inpw[1],color="red",label='inpw')
            ax.text(self.inpw[0],self.inpw[1],'inpw')
        ax.legend()
        self.fdf=fdf
        self.weightslist=weightslist
        self.cm=cm
        self.means=means
        return fig

        # %%
    def maxreturns(self,r):    
        #for a given risk r what is the maximum return one can get
        # r= float(input('Enter the fraction of risk, you have an apetite for: '))
        try:
            returnmax= max(self.fdf[(self.fdf["vars"]>r-0.001) & (self.fdf["vars"]<r+0.001)].means)
            print(f'The maximum return achievable for the risk apetite informed is {returnmax}')
            wei=self.weightslist[self.fdf[self.fdf.means==returnmax].index[0]]
            valuest=str(returnmax)+'\n and the corresponding weight proportions are \n'+str(wei)
            return valuest
        except:
            return 'Risk entered is not feasible'
        

