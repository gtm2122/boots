from xgboost import XGBClassifier
class clf_boots(object):
    def __init__(self,X,y,model=XGBClassifier,n=1000):
        self.model = model(nthread=-1)
        self.X = X
        self.y = y
        self.n = n
        self.model_list = []
    
    @staticmethod
    def boots(n,sz):
        inds=[]
        
        for i in range(10,n):
            ind = np.random.choice(sz,i)
            inds+=[ind]
            #print(ind.max())
        return inds
        
    def train(self,x_train,y_train):
        self.model.fit(x_train,y_train)
        return self.model
    
    def train_models(self):
        n = self.n
        sz = self.y.shape[0]
        boot_ind = self.boots(n,sz)
        for i in boot_ind:
            #print(i)
            self.model_list.append(self.train(self.X[np.array(i),:],self.y[np.array(i)]))
        #return 
        
    def forward(self,x):
        y = np.zeros((len(self.model_list),x.shape[0]))
        count = 0
        probs = np.zeros_like(y)
        for i in self.model_list:
            y[count,:] = i.predict(x)
            #print(i.predict_proba(x))
            #print(i.predict_proba(x)[:,1])
            probs[count,:] = i.predict_proba(x)[:,1]
            count+=1
        return np.round(y.sum(axis=0).astype(float)/y.shape[0]),np.mean(probs,axis=0)
#     def f_test(self,x):
#         print(len(self.model_list))
#         print(self.model_list[-1].predict(x))
    
    def test(self,x_test):
        ### Returns predictions and the prediction probabilities of positives
        return self.forward(x_test)[0],self.forward(x_test)[1]
