from filterPredictor import *
MSE=lambda output,target:(output-target)**2
MAPE=lambda output,target:abs(output-target)/output
def findBestWin(w,order,trainData):
    Loss=0
    w1=w+1
    for i in range(len(trainData)-w1):
        input=trainData[i:i+w]
        target=(trainData[i+w]+trainData[i+w1])/2
        output,residuals=predictorByPolynomial(input,order,len(input)+0.5)

        #Loss+=MSE(output,target)
        Loss+=MAPE(output,target)
    return Loss/(len(trainData)-w1)

if __name__=="__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    inputAdress="./data/Analysis/jpy.csv"
    outputAdress="./data/weight/w%d.txt"

    #data = np.loadtxt(open(inputAdress,"r"),delimiter="\t",skiprows=1,usecols=[1,3]) .T
    data = np.loadtxt(open(inputAdress,"r"),delimiter="\t",skiprows=1,usecols=[1,2]) .T#jpy缺少部分第三列
    trainData=filter(data[0],size=5,sigma=1)#buy

    x=np.arange(2,36)

    plt.subplot(1,2,1)

    RE=lambda x:1 if x >0 else -1
    W1=[findBestWin(w,1,trainData) for w in x]
    k1=[RE(i-j) for i,j in zip(W1[1:],W1[:-1])]
    plt.plot(x[:-1]+0.5,k1,"o-",label="1")
    #plt.plot(x,W1,"o-",label="2")

    W2=[findBestWin(w,2,trainData) for w in x[1:]]
    k2=[RE(i-j) for i,j in zip(W2[1:],W2[:-1])]
    plt.plot(x[1:-1]+0.5,k2,"o-",label="2")

    W3=[findBestWin(w,3,trainData) for w in x[2:]]
    k3=[RE(i-j) for i,j in zip(W3[1:],W3[:-1])]
    plt.plot(x[2:-1]+0.5,k3,"o-",label="3")
    plt.legend()
    plt.grid()
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    
    plt.subplot(1,2,2)
    plt.plot(x,W1,"o-",label="1")
    plt.plot(x[1:],W2,"o-",label="2")
    plt.plot(x[2:],W3,"o-",label="3")

    plt.legend()
    plt.grid()
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.show()