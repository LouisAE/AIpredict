from asyncio.windows_events import NULL
from symbol import parameters
import numpy as np
import cv2
np.set_printoptions(suppress=True)

class NN:
    def __init__(self,layer:list,LR:float):
        self.W=[]
        for t in range(len(layer)-1):
            w=np.random.normal(0.0,layer[t+1]**-0.5,(layer[t],layer[t+1]))
            #w=np.zeros((layer[t],layer[t+1]))+1/(layer[t]*layer[t+1])
            #w=np.zeros((layer[t],layer[t+1]))
            self.W.append(w)

        self.lr=LR
        #self.af=lambda x:1/(1+np.exp(0-x))
        self.af=lambda x:np.maximum(0,x)
        self.aff=lambda x:(x>0).astype(int)
        #self.af=lambda x:x
        #self.aff=lambda x:1
        pass
    
    def train(self,I,T):
        I=I.reshape((1,-1))
        T=T.reshape((1,-1))
        W=self.W
        H=[I,T]

        #forward
        for i in range(len(W)):
            H.insert(-1,self.af(H[i]@W[i]))

        #error
        Error=[H[-1]-H[-2]]
        for i in range(1,len(W)):
            Error.insert(0,(Error[0])@(W[len(W)-i].T))

        #backward
        for i in range(len(W)):
            #self.W[i]+=self.lr*(H[i].T@(Error[i]*H[i+1]*(1.0-H[i+1])))
            self.W[i]+=self.lr*(H[i].T@(Error[i]*self.aff(H[i+1])))

        #loss
        self.loss=(sum(Error[-1]**2)/len(Error[-1]))[0]
        pass
    
    def query(self,I):
        H=np.array(I@self.W[0])
        for t in range(1,len(self.W)):
            H=self.af(H@self.W[t])
        return H

def filter(L_Data,size=5,sigma=1):
    kernel=cv2.getGaussianKernel(2*size-1,sigma)[:size]
    kernel=np.array(kernel/np.sum(kernel)).flatten()

    k=(size-1)//2
    px=L_Data.shape[0]

    padding=np.zeros(px+size-1)#padding
    padding[k:px+k]=L_Data
    padding[:k]=L_Data[1:k+1][::-1]
    padding[-1:-k-1:-1]=L_Data[-k-1:-1]
    return np.around(np.correlate(padding,kernel),decimals=2)
    
def predictorByLinear(L_Data,predict:float=NULL):
    if predict==NULL:
        predict=len(L_Data)
    x=np.arange(len(L_Data))
    A=np.vstack([x, np.ones(len(x))]).T
    [k,b],residuals=np.linalg.lstsq(A,L_Data,rcond=None)[:2]
    return k*predict+b,residuals

def predictorByPolynomial(L_Data,order:int=3,predict:float=NULL):
    if predict==NULL:
        predict=len(L_Data)
    x=np.arange(len(L_Data))
    parameters,residuals=np.polyfit(x,L_Data,order,full=True)[:2]
    return np.poly1d(parameters)(predict),residuals
    #return a*len(L_Data)**3+b*len(L_Data)**2+c*len(L_Data)+d,R

#def predictorByNN(L_Data,W,af=lambda x:1/(1+np.exp(0-x))):
def predictorByNN(L_Data,W,af=lambda x:np.maximum(0,x)):
    H=np.array(L_Data@W[0])
    for t in range(1,len(W)):
        H=af(H@W[t])
    return H

def produceW(data,outputAdress):
    layer=[7,4,1]
    LR=0.01
    rounds=20
    n=NN(layer,LR)
    trainData=data.copy()

    error=[]
    for rounds in range(rounds):
        for i in range(len(trainData)-8):
            n.train(trainData[i:i+7],(trainData[i+7]+trainData[i+8])/2)#train
            #n.train(trainData[i:i+7],trainData[i+7])#train
            if i%50==0:
                error.append(n.query(trainData[i:i+7])-(trainData[i+7]+trainData[i+8])/2)
    
    for t in range(len(n.W)):
        np.savetxt(outputAdress%t,n.W[t])

    return n,error
if __name__=="__main__":
    import matplotlib.pyplot as plt
    inputAdress="./data/Analysis/usd.csv"
    outputAdress="./data/weight/w%d.txt"

    data = np.loadtxt(open(inputAdress,"r"),delimiter="\t",skiprows=1,usecols=[1,3]) .T
    trainData=filter(data[0],size=5,sigma=1)#buy
    #trainData=(trainData-np.mean(trainData))/np.std(trainData)#orthonormal
    #trainData=(trainData-np.min(trainData))/(np.max(trainData)-np.min(trainData))

    n,error=produceW(trainData,outputAdress)
    plt.subplot(2,1,1),plt.plot(np.arange(len(error)),error),plt.title("Training Errror")

    Y=trainData.copy()
    for i in range(len(Y)-9):
        #Y[i+8]=n.query(Y[i:i+7])
        Y[i+8]=predictorByNN(Y[i:i+7],n.W)
    plt.subplot(2,1,2)
    plt.plot(np.arange(len(Y)),Y,label="Prediction by NN")
    plt.plot(np.arange(len(trainData)),trainData,label="trainData")
    plt.title("Origin & Prediction")
    plt.legend()
    plt.show()
    plt.pause(0)
