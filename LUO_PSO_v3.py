## land use optimization of cultivated land and forest land
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
##    print time.ctime()
    out_array = np.ones((len(in_array),len(in_array[0]),4))
    for row in range(len(in_array)):
        for col in range(len(in_array[row])):
            out_array[row,col] = color_map[in_array[row,col]]
##    print time.ctime()
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
def adaptive_value(in_array,type_list,function_maps,nodata_value,weights):
    shape_index = shape_index_all(in_array,type_list,nodata_value,window_size = 5)
    out_value = shape_index*weights['S']
    func_values = values_calculate(in_array,function_maps,nodata_value)
    for f in func_values:
        out_value += weights[f[0]]*func_values[f]
    return out_value

    
class particle:

    def __init__(self,in_array,type_list,nodata_value,function_maps,init = 0.5,cognitive=0.5,social=0.5,weights={'F':0,'E':0,'C':1,'S':0}):
        self.array = in_array.copy()
        self.weights = weights
## template
        self.value = adaptive_value(in_array,type_list,function_maps,nodata_value,self.weights)
        self.P = in_array.copy()
        self.P_value = self.value
        self.cognitive = cognitive
        self.social = social
        self.init = init
        self.t = 1
        self.nodata_value = nodata_value  
    def update(self,group,target_areas_dict,function_maps,max_t=1000.0,area_balance = False):
##        print (time.ctime())
        if self.t > max_t:
            print ('reach maximum')
        else:
            group_value = [x.P_value for x in group]
            g_index = group_value.index(max([x for x in group_value]))
            g_array = group[g_index].P
            a,b,c,d=self.init,float(self.t)/max_t,self.cognitive,self.social
            rand_direction = random.choice(type_list)
            for row in range(len(self.array)):
                for col in range(len(self.array[row])):
                    if self.array[row,col] in target_areas_dict:
                        rand_num = random.uniform(0,a+b+c+d)
##                        neighbor = self.array[max(row-1,0):min(row+1,len(self.array)),max(col-1,0):min(col+1,len(self.array[0]))]
##                        if np.sum(neighbor!=self.array[row,col]) == 0:
##                            continue
                        if rand_num < a:
##                            self.array[row,col] = max_find(self.array[max([row-3,0]):min([row+3,len(self.array)]),max([col-3,0]):min([col+3,len(self.array[row])])],
##                                                           list(target_areas_dict.keys()))
                            self.array[row,col] = rand_direction
                        elif rand_num < a+b:
                            continue
                        elif rand_num < a+b+c:
                            self.array[row,col] = self.P[row,col]
                        else:
                            self.array[row,col] = g_array[row,col]                    
    #### area balance
            if area_balance:
                area_balance_p = {}
                area_balance_m = {}
                for ty in target_areas_dict:
                    area_balance = np.sum(self.array==ty)-target_areas_dict[ty]
                    if area_balance <0:
                        area_balance_m[ty] = -area_balance
                    else:
                        area_balance_p[ty] = area_balance
    ##            print(area_balance_p)
    ##            print(area_balance_m)
                for row in range(len(self.array)):
                    for col in range(len(self.array[row])):
                        or_ty = self.array[row,col]
                        if or_ty in area_balance_p and random.random()<float(area_balance_p[or_ty])/(np.sum(self.array[row+1:]==or_ty)+np.sum(self.array[row][col:]==or_ty)):
                            neighbor = self.array[max(row-1,0):min(row+1,len(self.array)),max(col-1,0):min(col+1,len(self.array[0]))]
                            if np.sum(neighbor == or_ty) != np.sum(neighbor!=self.nodata_value):
                                rand_num = random.uniform(0,sum(area_balance_m.values()))
                                t_num = 0
                                for ty in area_balance_m:
                                    t_num += area_balance_m[ty]
                                    if rand_num <= t_num:
                                        self.array[row,col] = ty
                                        area_balance_p[or_ty] -= 1
                                        area_balance_m[ty] -= 1
                                        break
                                
            
            new_value = adaptive_value(self.array,target_areas_dict.keys(),function_maps,self.nodata_value,self.weights)
            if new_value > self.P_value:
                self.P = self.array.copy()
                self.P_value = new_value
            self.value = new_value
            self.t += 1
                
        
if __name__ == '__main__':
    plt.ion()
##    land_array = arcpy.RasterToNumPyArray(land_map)
    land_array_or = open('data/lucc2015.txt').readlines()
    nodata_value = int(land_array_or[5].split()[1])
    land_array_or = np.array([[int(x) for x in y.split()] for y in land_array_or[6:]])
    land_array = land_array_or.copy()
    
    Food_productive_array = open('data/pcrop_2010_beijing').readlines()
##    Normalization and dissolve
    Food_productive_array = Normalization(np.array([[float(x) for x in y.split()] for y in Food_productive_array[6:]]),
                                           land_array ==1,float(Food_productive_array[5].split()[1]))
    Food_productive_array = dissolve(Food_productive_array,10,'mean')
    Food_productive_1 = Food_productive_array
    Food_productive_2 = Food_productive_array*0.0
    Food_productive_3 = Food_productive_array*0.5
    Economic_value_array = open('data/dgzs_2012.txt').readlines()
##    Normalization and dissolve
    Economic_value_array = Normalization(np.array([[float(x) for x in y.split()] for y in Economic_value_array[6:]]),
                                           land_array ==1,float(Economic_value_array[5].split()[1]))
    Economic_value_array = dissolve(Economic_value_array,10,'mean')
    Economic_value_1 = Economic_value_array*0.3
    Economic_value_2 = Economic_value_array*0.1
    Economic_value_3 = Economic_value_array
    Carbon_sequestration_array = open('data/soca.txt').readlines()
##    Normalization and dissolve
    Carbon_sequestration_array = Normalization(np.array([[float(x) for x in y.split()] for y in Carbon_sequestration_array[6:]]),
                                           land_array ==1,float(Carbon_sequestration_array[5].split()[1]))
    Carbon_sequestration_array = dissolve(Carbon_sequestration_array,10,'mean')
    Carbon_sequestration_1 = Carbon_sequestration_array*0.5
    Carbon_sequestration_2 = Carbon_sequestration_array
    Carbon_sequestration_3 = Carbon_sequestration_array*0.3
    weights = {'F':0.3,'E':0.3,'C':0.3,'S':0.1}

    land_array = dissolve(land_array,10)

    focuszone = [0,len(land_array),0,len(land_array[0])]
    focuszone = [50,100,100,150]
##    focuszone = [50,100,50,100]

    function_maps = {'Food productive_1':Food_productive_1[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],
                     'Food productive_2':Food_productive_2[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],
                     'Food productive_3':Food_productive_3[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],
                     'Economic value_1':Economic_value_1[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],
                     'Economic value_2':Economic_value_2[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],
                     'Economic value_3':Economic_value_3[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],
                     'Carbon sequestration_1':Carbon_sequestration_1[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],
                     'Carbon sequestration_2':Carbon_sequestration_2[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],
                     'Carbon sequestration_3':Carbon_sequestration_3[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]]
                     }  

    max_t = 100
    num_particle = 100
    init,congitive,social = 0.05,0.8,0.1
    

    
## reclass and init (1: crop; 2:afforest; 3: facility agricultural)

    for i in range(len(land_array)):
        for j in range(len(land_array[i])):
            if land_array[i,j] == 1:
                land_array[i,j] = random.randrange(1,4)
            else:
                land_array[i,j] = nodata_value
## end reclass
                
    land_array_show = np.zeros((len(land_array),len(land_array[0]),4))
    

    HVP = float(focuszone[3]-focuszone[2])/(focuszone[1]-focuszone[0])

    t = 1.0
## t i the transparency of original land use map
    cmap = {1:[1,0.8,0.0,t],2:[0.1,1.0,0.1,t],3:[1.0,0.1,0.1,t],
            4:[0.1,0.1,1.0,t],5:[1.0,0.1,0.1,t],6:[0.5,0.5,0.5,t],
            nodata_value:[0.9,0.9,0.9,1.0]}
    type_name = {1:'Traditional agricultural',2:'Afforestation',3:'Facility agricultural'}
    type_list = [x for x in np.unique(land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]]) if x != nodata_value]
    area_list = [np.sum(land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]]==x) for x in type_list]
    area_list = [np.sum(land_array==1),np.sum(land_array==2),np.sum(land_array==3)]
    area = dict(zip(type_list,area_list))
    for i in range(len(land_array)):
        for j in range(len(land_array[0])):
            land_array_show[i,j] = cmap[land_array[i,j]]
## setup layerout
    map_row = max([int(round(len(type_list))*HVP),1])
    fig_row,fig_col = map_row+3*max([int(map_row/2),1]),len(type_list)*2
    fig = plt.figure('land use optimization',(6,6),120)
    ax1 = plt.subplot2grid((fig_row,fig_col),(0,0),colspan=int(fig_col/2),rowspan=map_row)
    ax1.set_title('Original Land use map')
##    ax1.set_ylim(focuszone[2],focuszone[3])
##    ax1.set_xlim(focuszone[0],focuszone[1])
    ax1.imshow(land_array_show[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],interpolation='nearest')
    ax2 = plt.subplot2grid((fig_row,fig_col),(0,int(fig_col/2)),colspan=int(fig_col/2),rowspan=map_row)
    ax2.set_title('Optimized Land use map')
##    ax2.set_ylim(focuszone[2],focuszone[3])
##    ax2.set_xlim(focuszone[0],focuszone[1])
    ax2.imshow(land_array_show[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],interpolation='nearest')
    ax3 = plt.subplot2grid((fig_row,fig_col),(map_row,0),colspan=int(fig_col/2),rowspan=max([int(map_row/2),1]))   
    ax3.set_title('Values')
    ax3.get_yaxis().get_major_formatter().set_useOffset(False)
    ax4 = plt.subplot2grid((fig_row,fig_col),(map_row,int(fig_col/2)),colspan=int(fig_col/2),rowspan=max([int(map_row/2),1]))
    ax4.get_yaxis().get_major_formatter().set_useOffset(False)
    ax4.set_title('Functions')
    plt.tight_layout()
    plt.pause(0.01)

 
    
#### PSO simulating
    g_array = land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]]
    area_list_show = [area_sta(g_array,type_list)]
    SA_list_show = [shape_index(g_array,type_list,nodata_value)]
    function_list_show = [values_calculate(g_array,function_maps,nodata_value)]


    particle_group = [0]*num_particle
    target_areas_dict = area
    value_list = np.zeros((num_particle,max_t))
    P_values = [0]*max_t
    for i in range(num_particle):
##    ## reclass and init (1: crop; 2:afforest; 3: facility agricultural)
##        for r in range(len(land_array)):
##            for c in range(len(land_array[i])):
##                if land_array[r,c] != nodata_value:
##                    land_array[r,c] = random.randrange(1,4)
##                else:
##                    land_array[r,c] = nodata_value
##    ## end reclass
        particle_group[i] = (particle(land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],type_list,nodata_value,
                                      function_maps,init,congitive,social,weights = weights))
        value_list[i,0] = particle_group[i].value
    P_values[0] = max([x.P_value for x in particle_group])
    for t in range(1,max_t):
        for i in range(len(particle_group)):
            particle_group[i].update(particle_group,target_areas_dict,function_maps,max_t,weights)
            value_list[i,t] = particle_group[i].value

        group_value = [x.P_value for x in particle_group]
##        group_value = [x.value for x in particle_group]
        g_index = group_value.index(max([x for x in group_value]))
        g_array = particle_group[g_index].P
        ax2.imshow(array_show(g_array,cmap),interpolation='nearest')
        ax2.set_title('global optimized particle of interation {0}, value = {1}, particle_num: {2}'.format(t,round(particle_group[g_index].P_value,4),g_index))
        ax3.plot(np.array([x[:t+1] for x in value_list]).transpose(),color='0.3')
        P_values[t] = np.max(value_list)
        ax3.plot(P_values[:t+1],'r-',linewidth=2)
#### end PSO

#### calculate and draw function maps
        function_list = values_calculate(g_array,function_maps,nodata_value)
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
        SA_list_show.append(shape_index(g_array,type_list,nodata_value))
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
    
    plt.show(block=True)            
            
    

    
