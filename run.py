import numpy as np
import copy
import random
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics,datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import pandas as pd
from multiprocessing import Pool,cpu_count

class Dust:
    # flag indicate is a center
    def __init__(self, location, mass, flag, group):
        self.location = location
        self.mass = mass
        self.flag = flag
        self.group = group

gene_name_path = "./gene_name.txt"
gene_name_dict = {}
lines = open(gene_name_path,"r").readlines()
for line in lines:
    eid,gname = line.strip().split(",")
    gene_name_dict[eid]=gname

def load_TCGA(path):
    df = pd.read_csv(path,delimiter="\t",index_col=0)
    xx_eid = sorted(list(set(list(df.index))&set(list(gene_name_dict.keys()))))
    new_df = df.loc[xx_eid,]

    new_df.index = [gene_name_dict[xx] for xx in new_df.index]
    df = new_df

    tumor_list = []
    normal_list = []
    hnames = list(df.columns.values)
    for name in hnames:
        if name.endswith("-11"):
            normal_list.append(name)
        else:
            tumor_list.append(name)

    tumor_df = df[tumor_list]
    normal_df = df[normal_list]
    ids = list(df.index)

    '''
           g1,g2,g3,...,gn
    sample
    '''

    tumor_data = tumor_df.values.T
    normal_data = normal_df.values.T

    print("tumor num",len(tumor_data),"normal num",len(normal_data))

    data = np.concatenate([tumor_data,normal_data],axis=0)
    label = [0]*len(tumor_data)+[1]*len(normal_data)
    return data,label,ids



path = "./data/TCGA-HNSC-Merge_RNA_seq_FPKM-p-n.txt"      #19214 502 44
## path = "./data/TCGA-KIRC-Merge_RNA_seq_FPKM-p-n.txt"      #19214 531 72
## path = "./data/TCGA-KIRP-Merge_RNA_seq_FPKM-p-n.txt"      #19214 289 32
# path = "./data/TCGA-LIHC-Merge_RNA_seq_FPKM-p-n.txt"      #19214 373 50
# path = "./data/TCGA-LUAD-Merge_RNA_seq_FPKM-p-n.txt"      #19214 515 59
# path = "./data/TCGA-LUSC_Merge_RNA_seq_FPKM-p-n.txt"      #19214 501 49
# path = "./data/TCGA-PRAD-Merge_RNA_seq_FPKM-p-n.txt"        #19214 496 52
# path = "./data/TCGA-STAD-Merge_RNA_seq_FPKM-p-n.txt"      #19214 375 32
# path = "./data/TCGA-THCA-Merge_RNA_seq_FPKM-p-n.txt"      #19214 510 50
# path = "./data/TCGA-UCEC-Merge_RNA_seq_FPKM-p-n.txt"      #19214 544 35


data,label,ids = load_TCGA(path=path)
dim = data.shape[-1]
kdim = 50

seed = 1
data, inde_data = train_test_split(data, test_size=0.2, random_state=seed, shuffle=True)
label, inde_label = train_test_split(label, test_size=0.2, random_state=seed, shuffle=True)


def runx(location):
    train_data_x = data[:, location]
    train_label = label
    seed = random.choice(list(range(10000000)))
    rf = RandomForestClassifier(random_state=seed)
    rf.fit(train_data_x, train_label)
    imps = rf.feature_importances_
    return imps

pool = Pool(cpu_count()+10)


def generate_featuresx():
    impss = {i: 0 for i in range(dim)}
    num = 4000
    locations = [np.array(random.sample(list(range(dim)), kdim)).astype(int) for _ in range(num)]
    imps = pool.map(runx, locations)
    for i in range(len(imps)):
        for j in range(kdim):
            impss[locations[i][j]] = impss[locations[i][j]] + imps[i][j]
    impss_sorted = {id: val for id, val in sorted(impss.items(), key=lambda x: x[1], reverse=True)}
    tmp = list(impss_sorted.keys())
    return tmp

tmp_rank = generate_featuresx()
def generate_features():
    res = random.sample(tmp_rank[0:300],kdim)
    return res

def train_evl(x):
    seed = random.choice(list(range(100000)))
    train_data, test_data  = train_test_split(data,test_size=0.4,random_state=seed,shuffle=True)
    train_label,test_label = train_test_split(label,test_size=0.4,random_state=seed,shuffle=True)

    train_data_x = train_data[:,x]
    test_data_x = test_data[:,x]

    clf = SVC(probability=True)
    clf.fit(train_data_x,train_label)
    pred_label = clf.predict(test_data_x)
    pred_res = clf.predict_proba(test_data_x)[:,1]
    acc =metrics.accuracy_score(pred_label,test_label)
    ap = metrics.average_precision_score(y_true=test_label,y_score=pred_res)
    auc = metrics.roc_auc_score(y_true=test_label,y_score=pred_res)
    mcc = metrics.matthews_corrcoef(y_true=test_label,y_pred=pred_label)
    # print(ap,mcc)
    return np.exp(mcc)

def train_evlx(x):
    seed = random.choice(list(range(100000)))
    train_data, test_data  = train_test_split(data,test_size=0.4,random_state=seed,shuffle=True)
    train_label,test_label = train_test_split(label,test_size=0.4,random_state=seed,shuffle=True)

    train_data_x = train_data[:,x]
    test_data_x = test_data[:,x]

    clf = SVC(probability=True)
    clf.fit(train_data_x,train_label)
    pred_label = clf.predict(test_data_x)
    pred_res = clf.predict_proba(test_data_x)[:,1]
    acc =metrics.accuracy_score(pred_label,test_label)
    f1 = metrics.f1_score(y_true=test_label,y_pred=pred_label)
    recall = metrics.recall_score(y_true=test_label,y_pred=pred_label)
    precise = metrics.precision_score(y_true=test_label,y_pred=pred_label)
    ap = metrics.average_precision_score(y_true=test_label,y_score=pred_res)
    auc = metrics.roc_auc_score(y_true=test_label,y_score=pred_res)
    mcc = metrics.matthews_corrcoef(y_true=test_label,y_pred=pred_label)
    # print(ap,mcc)
    return acc,f1,recall,precise,mcc,ap,auc

def train_evl_test(x):

    train_data_x = data[:, x]
    test_data_x = inde_data[:, x]

    train_label = label
    test_label = inde_label

    clf = SVC(probability=True)
    clf.fit(train_data_x,train_label)
    pred_label = clf.predict(test_data_x)
    pred_res = clf.predict_proba(test_data_x)[:,1]
    acc =metrics.accuracy_score(pred_label,test_label)
    f1 = metrics.f1_score(y_true=test_label,y_pred=pred_label)
    recall = metrics.recall_score(y_true=test_label,y_pred=pred_label)
    precise = metrics.precision_score(y_true=test_label,y_pred=pred_label)
    ap = metrics.average_precision_score(y_true=test_label,y_score=pred_res)
    auc = metrics.roc_auc_score(y_true=test_label,y_score=pred_res)
    mcc = metrics.matthews_corrcoef(y_true=test_label,y_pred=pred_label)
    # print(ap,mcc)
    return acc,f1,recall,precise,mcc,ap,auc

def massFun(x):
    epoch = 10
    res = 0
    for _ in range(epoch):
        res = res + train_evl(x)
    tmp = np.exp(res/epoch)
    return tmp

class GFA:
    def __init__(self, dust_num=50, group_num=1):
        self.dust_num = dust_num
        self.group_num = group_num
        self.population = []
        self.center = []

    def initlize(self):
        dust_population = []
        for i in range(self.dust_num):
            location = generate_features()
            mass = massFun(location)
            group = np.random.randint(0, self.group_num)  
            flag = 0
            dust_i = Dust(location, mass, flag, group)
            dust_population.append(dust_i)
        self.population = dust_population
        self.getMaxDust()

    def getMaxDust(self):
        for i in range(len(self.population)):
            self.population[i].flag = 0
        population = self.population
        index = np.argmax([population[i].mass for i in range(len(population))])
        self.population[index].flag =1
        ppx = self.population[index].location
        random.shuffle(ppx)
        self.population[index].location = ppx
        self.center = self.population[index]

    def dis2(self,x,c):
        for _ in range(1):
            index = random.choice(list(range(len(c))))
            x[index] = c[index]
        return x

    def moveAndRotate(self):
        self.getMaxDust()
        population = self.population
        for i in range(len(population)):
            if population[i].flag == 0:
                population[i].location= self.dis2(population[i].location,self.center.location)
                pxx = list(population[i].location)
                random.shuffle(pxx)
                population[i].location = np.array([x if x>0 and x<dim else random.choice(list(range(dim))) for x in pxx ])
                population[i].mass = massFun(population[i].location)
        self.population = population

    def dustSort(self):
        self.population.sort(key=lambda dust: dust.mass, reverse=True)

    def absorb(self):
        population = self.population
        mean_mass = int(np.percentile([x.mass for x in population],q = 80))

        new_population = []
        for x in population:
            if x.mass>mean_mass:
                new_population.append(x)
        self.population = new_population
        self.getMaxDust()

    def explode(self):
        current_len = len(self.population)
        population = self.population
        indexs = random.sample(list(range(len(population))),int(len(population)*0.1))
        for index in indexs:
            for _ in range(int(kdim*0.1)):
                population[index].location[random.choice(list(range(len(population[index].location))))] = random.choice(list(range(dim)))
            population[index].mass = massFun(population[index].location)
        self.population = population

        for i in range(self.dust_num-current_len):
            # location = np.array(random.sample(list(range(dim)), kdim)).astype(int)
            location = generate_features()
            mass = massFun(location)
            group = np.random.randint(0, self.group_num) 
            flag = 0
            dust = Dust(location, mass, flag, group)
            self.population.append(dust)
        self.getMaxDust()

def get_gfa(dust_num,epoch):
    gfa = GFA(dust_num=dust_num)
    gfa.initlize()
    gfa.getMaxDust()
    for i in range(epoch):
        gfa.moveAndRotate()
        gfa.absorb()
        gfa.explode()
        x = gfa.center.location
        acc,f1,recall,precise,mcc,ap,auc= train_evlx(x)
        print(i)
        print(list(sorted(x)))
        print("acc",acc,"f1",f1,"precise",precise,"mcc",mcc,"ap",ap,"auc",auc)
    return gfa.center.location

if __name__=='__main__':
    dus_num = 50
    epoch = 50
    index = get_gfa(dust_num=dus_num,epoch=epoch)
    print(list(sorted(ids[x] for x in index)))
    acc,f1,recall,precise,mcc,ap,auc = train_evl_test(index)
    print(path)
    print("acc",acc,"f1",f1,"recall",recall,"precise",precise,"mcc",mcc,"ap",ap,"auc",auc)





