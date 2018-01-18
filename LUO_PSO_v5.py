## land use optimization of agricultural land
## adding utility particle
## ignore the carbon sequestration of plant
## encoding: utf-8
##
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = 6
import numpy as np
##arcpy.AddToolbox('C:/Users/yao/Documents/GitHub/Land_use optimization/ZonalMetrics-Toolbox-master/ZonalMetrics Toolbox/ZonalMetrics.pyt')
import os
import matplotlib.gridspec as gridspec
import random
import time
import math
##import thread
def array2point(array_list,nodata_value,seed):
    out_list = [(np.sum((x==1)),np.sum((x==2))) for x in array_list]
    return out_list
def weight_choice(type_list,weights):
    aa = np.random.uniform(0,np.sum(weights))
    a = 0
    for i in range(len(weights)):
        a += weights[i]
        if aa <= a:
            return type_list[i]
        else:
            continue
        
def entropy(array_list,type_list):
    out_res,count = 0.0,0.0
    for r in range(len(array_list[0])):
        for c in range(len(array_list[0][0])):
            if array_list[0][r,c] in type_list:
                for t in range(len(type_list)):
                    p = sum([x[r,c]==type_list[t] for x in array_list])/float(len(array_list))
                    if p != 0:
                        out_res += -p*math.log(p)
                count += 1
            else:
                continue
    return out_res/count
                
    
def dissolve(in_array,mix_num=5,method='max'):
    out_array = []
    for i in range(int(len(in_array)/mix_num)-1):
        out_array.append([])
        for j in range(int(len(in_array[i])/mix_num)-1):
            res = in_array[mix_num*i:mix_num*(i+1),mix_num*j:mix_num*(j+1)]
            if method == 'max':
                type_list = np.unique(res)
                out_array[i].append(max_find(res,type_list))
            elif method == 'mean':
                if np.sum(res>0) != 0:
                    out_array[i].append(np.sum(res*(res>0))/np.sum(res>0))
                else:
                    out_array[i].append(0)
    return np.array(out_array)
def Normalization(in_array,mask,nodata_value):
    out_array = in_array.copy()
    out_array = out_array*(out_array != nodata_value)*mask
    max_value = out_array.max()
    return out_array/max_value
    
    
def prob_choice(in_dict):
    rand_num = random.uniform(0,sum(in_dict.values()))
    start_num = 0
    for t in in_dict:
        start_num += in_dict[t]
        if rand_num < start_num:
            return t
def max_find(in_list,type_list):
    or_num = np.sum(in_list == type_list[0])
    or_ty = type_list[0]
    for x in type_list:
        num = np.sum(in_list==x)
        if num > or_num:
            or_num = num
            or_ty = x
    return or_ty
def array_show(in_array,color_map):
    out_array = np.ones((len(in_array),len(in_array[0]),4))
    for row in range(len(in_array)):
        for col in range(len(in_array[row])):
            out_array[row,col] = color_map[in_array[row,col]]

    return out_array

def area_sta(in_array,type_list):
    out_res = {}
    for t in type_list:
        out_res[t] = np.sum(in_array==t)
    return out_res

def shape_index(in_array,type_list,nodata_value=0,window_size=1):
    out_res = {}
    for t in type_list:
        out_res[t] = []
    for i in range(len(in_array)):
        for j in range(len(in_array[i])):
            if in_array[i,j] != nodata_value:
##                print i,j
                neighbor = in_array[max([0,i-window_size]):min([i+window_size+1,len(in_array)]),max([0,j-window_size]):min([j+window_size+1,len(in_array[0])])]
                out_res[in_array[i,j]].append(float(np.sum(neighbor==in_array[i,j]))/np.sum(neighbor != nodata_value))
    for t in type_list:
        out_res[t] = np.mean(np.array(out_res[t]))
    return out_res

def shape_index_all(in_array,type_list,nodata_value=0,window_size=1):
    out_res = 0
    num = 0
    for i in range(len(in_array)):
        for j in range(len(in_array[i])):
            if in_array[i,j] in type_list:
##                print i,j
                neighbor = in_array[max([0,i-window_size]):min([i+window_size+1,len(in_array)]),max([0,j-window_size]):min([j+window_size+1,len(in_array[0])])]
                out_res += float(np.sum(neighbor==in_array[i,j]))/np.sum(neighbor != nodata_value)
                num += 1
    return out_res/num

def values_calculate(in_array,value_arrays_dic,nodata_value):
## the format of the key of value_arrays_dic must be 'value_1'
## 1 is the code of related land use types
## the shape and spatial location of in_array and value arrays must same
    out_res = {}
    for value in value_arrays_dic.keys():
        if np.sum(in_array == int(value.split('_')[1])) != 0:
            try:
                out_res[value.split('_')[0]] += float(np.sum(value_arrays_dic[value]*(in_array==int(value.split('_')[1]))))/np.sum(in_array!=nodata_value)
            except:
                out_res[value.split('_')[0]] = float(np.sum(value_arrays_dic[value]*(in_array==int(value.split('_')[1]))))/np.sum(in_array!=nodata_value)
    return out_res
def adaptive_value(in_array,type_list,function_maps,nodata_value,weights,**kwargs):
    shape_index = shape_index_all(in_array,type_list,nodata_value,window_size = kwargs['window_size'])
    out_value = shape_index*weights['S']
    func_values = values_calculate(in_array,function_maps,nodata_value)
    for f in func_values:
        out_value += weights[f[0]]*func_values[f]
    return out_value

    
class particle:

    def __init__(self,in_array,type_list,nodata_value,function_maps,
                 init = 0.5,cognitive=0.5,social=0.5,
                 weights={'F':0,'E':0,'C':1,'S':0},**kwargs):
        try:
            self.window_size = kwargs['window_size']
        except:
            window_size = 1
        self.array = in_array.copy()
        self.weights = weights
## template
        self.value = adaptive_value(in_array,type_list,function_maps,nodata_value,self.weights,
                                    window_size = self.window_size)
        self.P = in_array.copy()
        self.P_value = self.value
        self.cognitive = cognitive
        self.social = social
        self.init = init
        self.t = 1.0
        self.nodata_value = nodata_value
    def update_zone(self,group,target_areas_dict,function_maps,max_t=1000.0,
                    area_balance = False,change_num = 5,change_size=0):
##        print (time.ctime())
        if self.t > max_t:
            print ('reach maximum')
        else:
            group_value = [x.P_value for x in group]
            g_index = group_value.index(max([x for x in group_value]))
            g_array = group[g_index].P
            a,b,c=self.init*(1-self.t/max_t),self.cognitive,self.social
            
            n = 1
            while True:
                if n > change_num:
                    break
                else:
                    row,col = random.randrange(0,len(self.array)),random.randrange(0,len(self.array[0]))
                    if self.array[row,col] in target_areas_dict:
                        rand_direction = weight_choice(list(target_areas_dict.keys()),np.random.uniform(0,1,len(target_areas_dict)))
                        rand_num = random.uniform(0,a+b+c)
                        for r in range(max(row-change_size,0),min(row+change_size,len(self.array))):
                            for c in range(max(col-change_size,0),min(col+change_size,len(self.array[0]))):
                                if self.array[r,c] in target_areas_dict:
                                    if rand_num < a:
                                        self.array[r,c] = rand_direction
                                    elif rand_num < a+b:
                                        self.array[r,c] = self.P[r,c]
                                    elif rand_num < a+b+c:
                                        self.array[r,c] = g_array[r,c]
                        n += 1
            
                                
            
            new_value = adaptive_value(self.array,target_areas_dict.keys(),function_maps,
                                       self.nodata_value,self.weights,window_size=self.window_size)
            if new_value > self.P_value:
                self.P = self.array.copy()
                self.P_value = new_value
            self.value = new_value
            self.t += 1
    def update_s(self,group,function_maps,type_list):
        self.array = get_sArray([x.P for x in group],type_list,self.nodata_value)
        new_value = adaptive_value(self.array,type_list,function_maps,
                                   self.nodata_value,self.weights,window_size=self.window_size)
        if new_value > self.P_value:
            self.P = self.array.copy()
            self.P_value = new_value
        self.value = new_value
        
        
        
def get_sArray(particle_group,type_list,nodata_value):
    out_array = np.ones((len(particle_group[0]),len(particle_group[0][0])))*nodata_value
    for i in range(len(out_array)):
        for j in range(len(out_array[0])):
            if particle_group[0][i,j] in type_list:
                out_array[i,j] = max_find([x[i,j] for x in particle_group],type_list)
            else:
                continue
    return out_array
        
        
        
        
def process(**kwargs):
    try:
        note = kwargs['note']
    except:
        note = 'None'
    print(time.ctime())
    plt.ion()
##    land_array = arcpy.RasterToNumPyArray(land_map)
    land_array_or = open('data/lucc2015.txt').readlines()
    nodata_value = int(land_array_or[5].split()[1])
    land_array_or = np.array([[int(x) for x in y.split()] for y in land_array_or[6:]])
    land_array = land_array_or.copy()
    
    Food_productive_array = open('data/pcrop_2010_beijing.txt').readlines()
##    Normalization and dissolve
    Food_productive_array = Normalization(np.array([[float(x) for x in y.split()] for y in Food_productive_array[6:]]),
                                           land_array ==1,float(Food_productive_array[5].split()[1]))
    Food_productive_array = dissolve(Food_productive_array,10,'mean')
    Food_productive_1 = Food_productive_array
    Food_productive_2 = Food_productive_array*0
    Food_productive_3 = Food_productive_array*0.7
    Carbon_sequestration_array = open('data/soca.txt').readlines()
    Max_SOCP = np.array([[float(x) for x in y.split()] for y in Carbon_sequestration_array[6:]]).max()
##    Normalization and dissolve
    Carbon_sequestration_array = Normalization(np.array([[float(x) for x in y.split()] for y in Carbon_sequestration_array[6:]]),
                                           land_array ==1,float(Carbon_sequestration_array[5].split()[1]))
    Carbon_sequestration_array = dissolve(Carbon_sequestration_array,10,'mean')
    Carbon_sequestration_1 = Carbon_sequestration_array*0
    Carbon_sequestration_2 = Carbon_sequestration_array
    Carbon_sequestration_3 = Carbon_sequestration_array*0.5

    try:
        weights = kwargs['weights']
    except:
        weights = {'F':0.25,'E':0.25,'C':0.25,'S':0.25}

    land_array = dissolve(land_array,10)
    type_list = [1,2,3]
    type_name = {1:'Traditional agricultural',2:'afforestation',3:'No-till farm'}
   

    focuszone = [0,len(land_array),0,len(land_array[0])]
 
##    focuszone = [100,200,100,150]
    land_array = land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]]


    function_maps = {'Food productive_1':Food_productive_1[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],
                     'Food productive_2':Food_productive_2[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],
                     'Food productive_3':Food_productive_3[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],
                     'Carbon sequestration_1':Carbon_sequestration_1[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],
                     'Carbon sequestration_2':Carbon_sequestration_2[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],
                     'Carbon sequestration_3':Carbon_sequestration_3[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]]
                     }  
    try:
        max_t,num_particle = kwargs['max_t'],kwargs['num_particle']
    except:
        max_t = 300
        num_particle = 100
    
    try:
        init,congitive,social = kwargs['init'],kwargs['congitive'],kwargs['social']
    except:
        init,congitive,social = 1,0.8,0.5
    try:
        window_size = kwargs['window_size']
    except:
        window_size = 1

    
## reclass and init (1: crop; 2:afforest; 3: facility agricultural)

    for i in range(len(land_array)):
        for j in range(len(land_array[i])):
             if land_array[i,j] == 1:
                land_array[i,j] = np.random.choice(type_list)
             else:
                land_array[i,j] = nodata_value
## end reclass
                
    land_array_show = np.zeros([np.shape(land_array)[0],np.shape(land_array)[1],4])
    

    HVP = float(focuszone[3]-focuszone[2])/(focuszone[1]-focuszone[0])

    t = 1.0
## t i the transparency of original land use map
    cmap = {1:[1,0.8,0.0,t],2:[0.1,1.0,0.1,t],3:[1.0,0.1,0.1,t],
            4:[0.1,0.1,1.0,t],5:[1.0,0.1,0.1,t],6:[0.5,0.5,0.5,t],
            nodata_value:[0.9,0.9,0.9,1.0]}
##    type_list = [x for x in np.unique(land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]]) if x != nodata_value]
    area_list = [np.sum(land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]]==x) for x in type_list]
    area_list = [np.sum(land_array==1),np.sum(land_array==2),np.sum(land_array==3)]
    area = dict(zip(type_list,area_list))
    for i in range(len(land_array)):
        for j in range(len(land_array[0])):
            land_array_show[i,j] = cmap[land_array[i,j]]
## setup layerout
    map_row = max([int(round(len(type_list))*HVP),1])
    fig_row,fig_col = map_row+3*max([int(map_row/2),1]),len(type_list)*2
    fig = plt.figure('land use optimization :{0}'.format(weights),(6,6),120)
    ax1 = plt.subplot2grid((fig_row,fig_col),(0,0),colspan=int(fig_col/2),rowspan=map_row)
    
    ax2 = plt.subplot2grid((fig_row,fig_col),(0,int(fig_col/2)),colspan=int(fig_col/2),rowspan=map_row)

##    ax2.imshow(land_array_show[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],interpolation='nearest')
    ax3 = plt.subplot2grid((fig_row,fig_col),(map_row,0),colspan=int(fig_col/2),rowspan=max([int(map_row/2),1]))   
##    ax3.set_title('Values')
    ax3.get_yaxis().get_major_formatter().set_useOffset(False)
    ax3t = ax3.twinx()
    ax3t.get_yaxis().get_major_formatter().set_useOffset(False)
    ax4 = plt.subplot2grid((fig_row,fig_col),(map_row,int(fig_col/2)),colspan=int(fig_col/2),rowspan=max([int(map_row/2),1]))
    ax4.get_yaxis().get_major_formatter().set_useOffset(False)
##    ax4.set_title('Functions')
    plt.tight_layout()

 
    
#### PSO simulating
    g_array = land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]]
    area_list_show = []
    SA_list_show = []
    function_list_show = []

    etp_list = [0]*(max_t)

    particle_group = [0]*num_particle
    target_areas_dict = area
    value_list = np.zeros((num_particle,max_t+1))
    P_values = [0]*(max_t+1)
    etp_list = [0]*(max_t+1)
    for i in range(num_particle):
        t_weights = np.random.uniform(0,1,len(type_list))
    ## reclass and init (1: crop; 2:afforest; 3: facility agricultural)
        for r in range(len(land_array)):
            for c in range(len(land_array[i])):
                if land_array[r,c] != nodata_value:                   
                    land_array[r,c] = np.random.choice(type_list)
                else:
                    land_array[r,c] = nodata_value
    ## end reclass
        
        particle_group[i] = (particle(land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],type_list,nodata_value,
                                      function_maps,init,congitive,social,weights = weights,window_size=window_size))
        value_list[i,0] = particle_group[i].value
    P_values[0] = max([x.P_value for x in particle_group])
    etp_list[0] = entropy([x.P for x in particle_group],type_list)
    g_points = []


    for t in range(0,max_t+1):
        print(time.ctime())
        for i in range(num_particle):
            if t > 0:
                particle_group[i].update_zone(particle_group,target_areas_dict,function_maps,
                                              max_t,weights,change_num=kwargs['change_num'],change_size=window_size)
            value_list[i,t] = particle_group[i].value

        group_value = [x.P_value for x in particle_group]
        g_index = group_value.index(max([x for x in group_value]))
        g_array = particle_group[g_index].P
        ax1.cla()
        xlabel,ylabel = np.unique([x.split('_')[0] for x in function_maps.keys()])
        point_values = [values_calculate(x.array,function_maps,nodata_value) for x in particle_group]
        points = [(x[xlabel],x[ylabel]) for x in point_values]
        gp_values = values_calculate(g_array,function_maps,nodata_value)
        g_points.append((gp_values[xlabel],gp_values[ylabel]))
        ax1.scatter([x[0] for x in points],[x[1] for x in points],s=1)
        for i in range(len(particle_group)):
            ax1.text(points[i][0],points[i][1],i)
        ax1.plot([x[0] for x in g_points],[x[1] for x in g_points],'r-')
        ax1.scatter(g_points[0][0],g_points[0][1])
        ax1.scatter(g_points[-1][0],g_points[-1][1],marker='x',color='r')
        ax1.set_title('particles in 2-D surface')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax2.cla()
        ax2.imshow(array_show(g_array,cmap),interpolation='nearest')
        ax2.set_title('Interation {0}, value = {1}, particle_num: {2}'.format(t,round(particle_group[g_index].P_value,4),g_index))
        ax3.cla()
        ax3t.cla()
        ax3.plot(np.array([x[:t+1]/P_values[0] for x in value_list]).transpose(),color='0.3',lw = 0.1)
        P_values[t] = np.max(value_list)
        etp_list[t] = entropy([x.P for x in particle_group],type_list)
        ax3.set_title('Values (entropy={0})'.format(round(etp_list[t],4)))
        ax3.plot(np.array(P_values[:t+1])/P_values[0],'r-',linewidth=2)
        ax3t.plot(etp_list[:t+1],'b-',lw=1)
        
#### end PSO

#### calculate and draw function maps
        function_list = values_calculate(g_array,function_maps,nodata_value)
        function_list['spatial compactness'] = shape_index_all(g_array,type_list,nodata_value,
                                                               window_size = window_size)
        function_list_show.append(function_list)
        ax4.cla()
        ax4.set_title('functions of optimized particle')
        ax4.get_yaxis().get_major_formatter().set_useOffset(False)
        for function in function_list.keys():
            ax4.plot([x[function]/function_list_show[0][function] for x in function_list_show],label = function)
        ax4.legend()
## calculate and draw area and shape index changes
        index = 0
        area_list_show.append(area_sta(g_array,type_list))                    
        SA_list_show.append(shape_index(g_array,type_list,nodata_value,window_size=window_size))
        for t in type_list:
            sub_ax1 = plt.subplot2grid((fig_row,fig_col),(map_row+max([int(map_row/2),1]),0+2*index),
                                       colspan=2,rowspan=max([int(map_row/2),1]))

            sub_ax1.get_yaxis().get_major_formatter().set_useOffset(False)
            sub_ax1.set_title('Area of {0}'.format(type_name[t]))
##                        plt.xlim(0,max_iteration/update_interval+1)
            sub_ax1.plot([x[t] for x in area_list_show],'-',label=type_name[t],color = cmap[t])
            sub_ax2 = plt.subplot2grid((fig_row,fig_col),(map_row+2*max([int(map_row/2),1]),0+2*index),
                                       colspan=2,rowspan=max([int(map_row/2),1]))
            sub_ax2.set_title('Shape index of {0}'.format(type_name[t]))
            sub_ax2.get_yaxis().get_major_formatter().set_useOffset(False)
            sub_ax2.plot([x[t] for x in SA_list_show],'-',label=type_name[t],color = cmap[t])
            index += 1
## update drawing
            
        plt.pause(0.001)
    record_num = 0
    while True:
        if os.path.exists('record_{0}.txt'.format(record_num)):
            record_num += 1
        else:
            record = open('record_{0}.txt'.format(record_num),'w+')
            record.write(note+'\n')
            record.write('time:{0}\nweight:{1}\nparticle_num:{2}\n'.format(time.ctime(),weights,num_particle))
            record.write('*****particle parameters******\n')
            record.write('init:{0}\ncongitive:{1}\nsocial:{2}\n'.format(init,congitive,social))
            record.write('window_size:{0}\n'.format(window_size))
            record.write('*****results******\n')
            record.write('Area:{0}\nValues:{1}\nValue:{2}'.format(area_list_show[-1],
                                                                  function_list_show[-1],
                                                                  P_values[-1]))
            record.close()
            print('save result with record_{0}.txt'.format(record_num))
            break
    plt.savefig('record_{0}.png'.format(record_num))
    
    plt.close()

if __name__ == '__main__':
    note = 'change the update method as zone,initilize particle with different statues\n'
    note = note + 'using the P_array of each particle to calculate the entropy of the swarm\n'
    note = note + 'adding special particle updated based on other particles'
    note = note + ', change_num = 200'
    weights_list = [{'F':0.3,'E':0.3,'C':0.3,'S':0.1},
                    {'F':0.4,'E':0.25,'C':0.25,'S':0.1},
                    {'F':0.25,'E':0.4,'C':0.25,'S':0.1},
                    {'F':0.25,'E':0.25,'C':0.4,'S':0.1}]
    weights_list = []
    for i in np.arange(0.1,0.9,0.1):
        weights_list.append({'F':round(i,2),'C':round(0.9-i,2),'S':0.1})
    
    for weights in weights_list:
        process(max_t=2000,num_particle = 50,window_size=1,note=note,change_num=200,
                init=1,congitive=0.8,social=0.1,weights = weights)
        
        ###
        
    
            
    

    
