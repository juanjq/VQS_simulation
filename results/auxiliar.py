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
    
def fidelity(psi,phi):
    return abs(np.vdot(psi,phi))


# calculating the number of variables we need to input for our case
def nVariables(N,ncycles,mode):
    
    if mode=='evenodd':
        var = int(ncycles*(3))
        
    elif mode == 'pairs':
        var = int(ncycles*(2*(N/2-N%2/2)+1))
        
    elif mode == 'pairs_nnn':
        var = int(ncycles*(2*(N/2-N%2/2)+1))
        
    elif mode == 'pairs_conc':
        var = int(ncycles*(N)) 
        
    elif mode == 'individual':
        var = int(ncycles*(N+1)) 
            
    return var


#write in a document   
def write(x,doc,path):
    file = open(path + doc + '.txt','a+')
    file.write(str(x))
    file.write('\n')
    file.close()
    
def sort(E,T):
    
    srt = sorted(zip(E,T),key=lambda x: x[0])
    E = [srt[i][0] for i in range(len(list(srt)))]
    T = [srt[i][1] for i in range(len(list(srt)))]

    return E,T    
        
    
def readData(option,path):
    
    with open(path + option + '.txt', 'r') as file:
        d = file.read().rstrip()            
        
    runs = d.split('new-run\n')[1:]
    
    runs_split = [runs[i].split('\n') for i in range(len(runs))]
    
    data = [[ runs_split[i][j].split(',') for j in range(len(runs_split[i]))][:-1] for i in range(len(runs_split))]
    data = [[[ float(data[i][j][k]) for k in range(len(data[i][j]))]for j in range(len(data[i]))] for i in range(len(data))]
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j][0]=int(data[i][j][0])
        
    print('Number of runs : '+str(len(data)))
    
    data = [np.transpose(data[i]) for i in range(len(data))]
    
    return data
