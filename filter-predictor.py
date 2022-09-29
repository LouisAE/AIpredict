from cProfile import label
import numpy as np
from math import ceil
import cv2

class NN:
    def __init__(self,layer:list,LR:float):
        self.W=[]
        for t in range(len(layer)-1):
            #w=np.random.normal(0.0,layer[t+1]**-0.5,(layer[t],layer[t+1]))
            #w=np.zeros((layer[t],layer[t+1]))+1/(layer[t]*layer[t+1])
            w=np.zeros((layer[t],layer[t+1]))
            self.W.append(w)

        self.lr=LR
        self.af=lambda x:1/(1+np.exp(0-x))
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
            self.W[i]+=self.lr*(H[i].T@(Error[i]*H[i+1]*(1.0-H[i+1])))

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
    return np.convolve(L_Data,kernel,mode='full')
    
def predictorByLinear(L_Data):
    x=np.arange(len(L_Data))
    [k,b],R=np.linalg.lstsq(x,L_Data,rcond=None)
    return k*len(L_Data)+b,R

def predictorByCube(L_Data):
    x=np.arange(len(L_Data))
    [a,b,c,d],R=np.polyfit(x,L_Data,3,full=True)
    return a*len(L_Data)**3+b*len(L_Data)**2+c*len(L_Data)+d,R

def predictorByNN(L_Data,W,af=lambda x:1/(1+np.exp(0-x))):
    H=np.array(L_Data@W[0])
    for t in range(1,len(W)):
        H=af(H@W[t])
    return H

def produceW(data,outputAdress):
    layer=[7,4,1]
    LR=0.01
    rounds=50
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

    data = np.loadtxt(open(inputAdress,"r"),delimiter="\t",skiprows=1,usecols=[0,2]) .T
    trainData=filter(data[0],size=5,sigma=1)#buy
    #trainData=(trainData-np.mean(trainData))/np.std(trainData)
    trainData=(trainData-np.min(trainData))/(np.max(trainData)-np.min(trainData))

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
