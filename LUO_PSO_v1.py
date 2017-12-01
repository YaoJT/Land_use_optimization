## land use optimization
## encoding: utf-8
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = 6
import numpy as np
import arcpy
##arcpy.AddToolbox('C:/Users/yao/Documents/GitHub/Land_use optimization/ZonalMetrics-Toolbox-master/ZonalMetrics Toolbox/ZonalMetrics.pyt')
import os
import matplotlib.gridspec as gridspec
import random

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

def shape_index_all(in_array,type_list,nodata_value=0,window_size=3):
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
            out_res[value] = float(np.sum(value_arrays_dic[value]*(in_array==int(value.split('_')[1]))))/np.sum(in_array == int(value.split('_')[1]))
    return out_res
def adaptive_value(in_array,type_list,function_maps,nodata_value):
    out_value = 0
    shape_indexs = shape_index(in_array,type_list,nodata_value,window_size = 1)
    func_values = values_calculate(in_array,function_maps,nodata_value)
    for func in func_values:
        out_value += func_values[func]*shape_indexs[int(func.split('_')[1])]
    return out_value

    
class particle:

    def __init__(self,in_array,type_list,nodata_value,function_maps,cognitive=0.5,social=0.5):
        self.array = in_array.copy()
## template
        self.value = adaptive_value(in_array,type_list,function_maps,nodata_value)
        self.P = in_array.copy()
        self.P_value = self.value
        self.cognitive = cognitive
        self.social = social
        self.t = 1
        self.nodata_value = nodata_value
    def update(self,group,target_areas_dict,function_maps,max_t=1000.0):
        print time.ctime()
        if self.t > max_t:
            print 'reach maximum'
        else:
            group_value = [x.P_value for x in group]
            g_index = group_value.index(max([x for x in group_value]))
            g_array = group[g_index].P
            a,b,c,d=0.01,float(self.t)/max_t,self.cognitive,self.social
            for row in range(len(self.array)):
                for col in range(len(self.array[row])):
                    if self.array[row,col] in target_areas_dict:
                        rand_num = random.uniform(0,a+b+c+d)
                        if rand_num < a:
                            self.array[row,col] = max_find(self.array[max([row-3,0]):min([row+3,len(self.array)]),max([col-3,0]):min([col+3,len(self.array[row])])],
                                                           target_areas_dict.keys())
                        elif rand_num < a+b:
                            self.array[row,col] = self.array[row,col]
                        elif rand_num < a+b+c:
                            self.array[row,col] = self.P[row,col]
                        else:
                            self.array[row,col] = g_array[row,col]                    
    #### area balance
            area_balance_p = {}
            area_balance_m = {}
            for ty in target_areas_dict:
                area_balance = np.sum(self.array==ty)-target_areas_dict[ty]
                if area_balance <0:
                    area_balance_m[ty] = -area_balance
                else:
                    area_balance_p[ty] = area_balance
            for row in range(len(self.array)):
                for col in range(len(self.array[row])):
                    or_ty = self.array[row,col]
                    if or_ty in area_balance_p and random.random()<float(area_balance_p[or_ty])/(np.sum(self.array[row+1:]==or_ty)+np.sum(self.array[row][col:]==or_ty)):
                        rand_num = random.uniform(0,sum(area_balance_m.values()))
                        t_num = 0
                        for ty in area_balance_m:
                            t_num += area_balance_m[ty]
                            if rand_num <= t_num:
                                self.array[row,col] = ty
                                area_balance_p[or_ty] -= 1
                                area_balance_m[ty] -= 1
                                break
            
            new_value = adaptive_value(self.array,target_areas_dict.keys(),function_maps,self.nodata_value)
            if new_value > self.P_value:
                self.P = self.array.copy()
                self.P_value = new_value
            self.value = new_value
            self.t += 1
                
        
if __name__ == '__main__':
    plt.ion()
    land_map = 'lucc2015.tif'
    SOCP_map = ''
    land_array = arcpy.RasterToNumPyArray(land_map)
    land_array_show = np.zeros((len(land_array),len(land_array[0]),4))
    nodata_value = arcpy.Raster(land_map).noDataValue
    focuszone = [1300,1400,300,500]
    HVP = float(focuszone[3]-focuszone[2])/(focuszone[1]-focuszone[0])
#### reclass of land use map
    for i in range(len(land_array)):
        for j in range(len(land_array[0])):
            if land_array[i,j] >10 and land_array[i,j]<100:
                land_array[i,j] = 5

    t = 1.0
## t i the transparency of original land use map
    cmap = {1:[1,0.8,0.0,t],2:[0.1,1.0,0.1,t],3:[0.7,1.0,0.2,t],
            4:[0.1,0.1,1.0,t],5:[1.0,0.1,0.1,t],6:[0.5,0.5,0.5,t],
            nodata_value:[0.9,0.9,0.9,1.0]}
    type_name = {1:'Agriculturla land',2:'Forest land',3:'Grass land',
                 4:'Water body',5:'Built-up land',6:'Other land'}
    type_list = [x for x in np.unique(land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]]) if x != nodata_value]
    area_list = [np.sum(land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]]==x) for x in type_list]
    area = dict(zip(type_list,area_list))
    for i in range(len(land_array)):
        for j in range(len(land_array[0])):
            land_array_show[i,j] = cmap[land_array[i,j]]
## setup layerout
    map_row = max([int(round(len(type_list))*HVP),1])
    fig_row,fig_col = map_row+3*max([(map_row/2),1]),len(type_list)*2
    fig = plt.figure('land use optimization',(6,6),120)
    ax1 = plt.subplot2grid((fig_row,fig_col),(0,0),colspan=fig_col/2,rowspan=map_row)
    ax1.set_title('Original Land use map')
##    ax1.set_ylim(focuszone[2],focuszone[3])
##    ax1.set_xlim(focuszone[0],focuszone[1])
    ax1.imshow(land_array_show[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],interpolation='nearest')
    ax2 = plt.subplot2grid((fig_row,fig_col),(0,fig_col/2),colspan=fig_col/2,rowspan=map_row)
    ax2.set_title('Optimized Land use map')
##    ax2.set_ylim(focuszone[2],focuszone[3])
##    ax2.set_xlim(focuszone[0],focuszone[1])
    ax2.imshow(land_array_show[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],interpolation='nearest')
    ax3 = plt.subplot2grid((fig_row,fig_col),(map_row,0),colspan=fig_col/2,rowspan=max([map_row/2,1]))   
    ax3.set_title('Values')
    ax3.get_yaxis().get_major_formatter().set_useOffset(False)
    ax4 = plt.subplot2grid((fig_row,fig_col),(map_row,fig_col/2),colspan=fig_col/2,rowspan=max([map_row/2,1]))
    ax4.get_yaxis().get_major_formatter().set_useOffset(False)
    ax4.set_title('Functions')
    plt.tight_layout()
    plt.pause(0.01)

## using the random array for template
    
    function_maps = {'SC_1':np.ones([focuszone[1]-focuszone[0],focuszone[3]-focuszone[2]]),
                     'SC_2':np.ones([focuszone[1]-focuszone[0],focuszone[3]-focuszone[2]]),
                     'SC_5':np.ones((focuszone[1]-focuszone[0],focuszone[3]-focuszone[2]))}
  
    g_array = land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]]
    area_list_show = [area_sta(g_array,type_list)]
    SA_list_show = [shape_index(g_array,type_list,nodata_value)]
    function_list_show = [values_calculate(g_array,function_maps,nodata_value)]
    

    
    max_t = 300
    num_particle = 10
    particle_group = [0]*num_particle
    target_areas_dict = area
    value_list = np.zeros((num_particle,max_t))
    P_values = [0]*max_t
    for i in range(num_particle):
        particle_group[i] = (particle(land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],type_list,nodata_value,function_maps))
        value_list[i,0] = particle_group[i].value
    P_values[0] = max([x.P_value for x in particle_group])
    for t in range(1,max_t):
        for i in range(len(particle_group)):
            particle_group[i].update(particle_group,target_areas_dict,function_maps,max_t)
            value_list[i,t] = particle_group[i].value
        group_value = [x.P_value for x in particle_group]
        g_index = group_value.index(max([x for x in group_value]))
        g_array = particle_group[g_index].P
        ax2.imshow(array_show(g_array,cmap),interpolation='nearest')
        ax2.set_title('global optimized particle of interation {0}, value = {1}, particle_num: {2}'.format(t,particle_group[g_index].P_value,g_index))
        ax3.plot(np.array([x[:t+1] for x in value_list]).transpose(),color='0.3')
        P_values[t] = np.max(value_list)
        ax3.plot(P_values[:t+1],'r-',linewidth=2)


#### calculate and draw function maps
        function_list = values_calculate(g_array,function_maps,nodata_value)
        function_list_show.append(function_list)
        ax4.cla()
        ax4.set_title('functions of optimized particle')
        ax4.get_yaxis().get_major_formatter().set_useOffset(False)
        for function in function_list.keys():
            ax4.plot([x[function] for x in function_list_show],label = function)
        ax4.legend()
## calculate and draw area and shape index changes
        index = 0
        area_list_show.append(area_sta(g_array,type_list))                    
        SA_list_show.append(shape_index(g_array,type_list,nodata_value))
        for t in type_list:
            sub_ax1 = plt.subplot2grid((fig_row,fig_col),(map_row+max([map_row/2,1]),0+2*index),colspan=2,rowspan=max([map_row/2,1]))

            sub_ax1.get_yaxis().get_major_formatter().set_useOffset(False)
            sub_ax1.set_title('Area of {0}'.format(type_name[t]))
##                        plt.xlim(0,max_iteration/update_interval+1)
            sub_ax1.plot([x[t] for x in area_list_show],'-',label=type_name[t],color = cmap[t])
            sub_ax2 = plt.subplot2grid((fig_row,fig_col),(map_row+2*max([map_row/2,1]),0+2*index),colspan=2,rowspan=max([map_row/2,1]))
            sub_ax2.set_title('Shape index of {0}'.format(type_name[t]))
            sub_ax2.get_yaxis().get_major_formatter().set_useOffset(False)
            sub_ax2.plot([x[t] for x in SA_list_show],'-',label=type_name[t],color = cmap[t])
            index += 1
## update drawing
        plt.pause(0.001)
##    plt.show(block=True)
            
            
    

    
