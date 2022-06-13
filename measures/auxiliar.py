import numpy as np

# SIMPLE FUNCTIONS #####################

# normalize vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm != 0:
        return v/norm
    else:
        return v

# calculate the braket
def braket(H,psi):
    return np.dot(np.dot(np.conj(psi),H),psi)
    
#write in a document   
def write(x,doc,path):
    
    file = open(path+doc+'.txt','a+')
    file.write(str(x))
    file.write('\n')
    file.close()
   
    
def readData(option,path):
    
    with open(path+option+'.txt', 'r') as file:
        data = file.read().rstrip()        
        
    runs = data.split('new-run\n')[1:]
    
    runs_split = [runs[i].split('\n') for i in range(len(runs))]
    
    data = [[ runs_split[i][j].split(',') for j in range(len(runs_split[i]))][:-1] for i in range(len(runs_split))]
    data = [[[ float(data[i][j][k]) for k in range(len(data[i][j]))]for j in range(len(data[i]))] for i in range(len(data))]
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j][0]=int(data[i][j][0])
                
    print('Number of runs : '+str(len(data)))
        
    data = [np.transpose(data[i]) for i in range(len(data))]
    
    return data