from xgboost import XGBClassifier
import numpy as np
### Boosted classifiers of different sized sampled(with replacement) subsets of dataset

class clf_boots2(object):
    def __init__(self,X,y,model=XGBClassifier,n=1000):
        self.model = model(nthread=-1)
        self.X = X
        self.y = y
        self.n = n
        self.model_list = []
    
    def la(self):
        nn = self.n
        for i in range(0,self.n+20):
            print(i)
    
    def this_boots(self):
        inds=[]
        #print('her')
        #print(self.n)
        for i in xrange(1,self.n):
            #print(i)
            #print('herr')
            num = (i+10)%self.n
	    ind = np.floor(np.random.rand(i)*self.y.shape[0]).astype(int)
            #ind = np.random.choice(self.y.shape[0],num)
            #print(np.random.choice(self.y.shape[0],i))
            #print(ind)
            #print(ind)
            inds+=[ind]
            #print(ind.max())
        return inds
        
    def train(self,x_train,y_train):
        self.model.fit(x_train,y_train)
        return self.model
    
    def train_models(self):
        #print(self.n)
        #print(self.y.shape[0])
        boot_ind = self.this_boots()
        #print(boot_ind)
        for i in boot_ind:
            #print(i)
            self.model_list.append(self.train(self.X[np.array(i),:],self.y[np.array(i)]))
        #return 
        
    def forward(self,x):
        y = np.zeros((len(self.model_list),x.shape[0]))
        #print(self.model_list)
        count = 0
        probs = np.zeros_like(y)
        #print(probs)
        for i in self.model_list:
            y[count,:] = i.predict(x)
            #print(i.predict_proba(x))
            #print(i.predict_proba(x)[:,1])
            probs[count,:] = i.predict_proba(x)[:,1]
            #print(probs[count,:])
            count+=1
        #print(probs)
        prob_result = np.zeros((y.shape[1],2))
        print np.mean(probs,axis=0)
        prob_result[:,1] = np.mean(probs,axis=0)
        prob_result[:,0] = 1 - prob_result[:,1]
        
        return np.round(y.sum(axis=0).astype(float)/y.shape[0]),prob_result
#     def f_test(self,x):
#         print(len(self.model_list))
#         print(self.model_list[-1].predict(x))
    
    def test(self,x_test):
        ### Returns predictions and the prediction probabilities of positives
        a,b = self.forward(x_test)
        return a,b  
 
