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

def area_sta(in_array,type_list):
    out_res = {}
    for t in type_list:
        out_res[t] = np.sum(in_array==t)
    return out_res

def shape_index(in_array,type_list,nodata_value=0,window_size=3):
    out_res = {}
    for t in type_list:
        out_res[t] = []
    for i in range(len(in_array)):
        for j in range(len(in_array[i])):
            if in_array[i,j] != nodata_value:
##                print i,j
                neighbor = in_array[max([0,i-window_size]):min([i+window_size,len(in_array)]),max([0,j-window_size]):min([j+window_size,len(in_array[0])])]
                out_res[in_array[i,j]].append(float(np.sum(neighbor==in_array[i,j]))/np.sum(neighbor != nodata_value))
    for t in type_list:
        out_res[t] = np.mean(np.array(out_res[t]))
    return out_res

def values_calculate(in_array,value_arrays_dic,nodata_value):
## the format of the key of value_arrays_dic must be 'value_1'
## 1 is the code of related land use types
## the shape and spatial location of in_array and value arrays must same
    out_res = {}
    for value in value_arrays_dic.keys():
        if np.sum(in_array == int(value.split('_')[1])) != 0:
            out_res[value] = float(np.sum(value_arrays_dic[value]*(in_array==int(value.split('_')[1]))))/np.sum(in_array == int(value.split('_')[1]))
    return out_res
def adaptive_value(in_array,type_list,value_arraus):

    return shape_index(in_array,type_list)
    
class particle:

    def __init__(self,in_array,type_list,cognitive=0.5,social=0.5):
        self.array = array
## template
        self.value = adaptive_value(array,type_list,value_arrays={})
        self.P = in_array
        self.P_value = self.value
        self.cognitive = cognitive
        self.social = social
    def update(self,group,t=0,max_t=1000.0):
        g_index = group.index(max[x.value for x in group])
        g_array = group[g_index].array
        a,b,c,d=1-float(t)/max_t,float(t)/max_t,self.cognitive,self.social
        for row in len(self.array):
            for col in len(self.array[row]):
                if self.array[row,col] in type_list:
                    rand_num = random.uniform(0,a+b+c+d)
                    if rand_num < a:
                        self.array[row,col] = random(type_list)
                    elif rand_num < a+b:
                        self.array[row,col] = self.array[row,col]
                    elif rand_num < a+b+c:
                        self.array[row,col] = self.P[row,col]
                    else:
                        self.array[row,col] = g_array[row,col]
#### area balance
                    
                    
                
        
if __name__ == '__main__':
    plt.ion()
    land_map = 'lucc2015.tif'
    land_array = arcpy.RasterToNumPyArray(land_map)
    land_array_show = np.zeros((len(land_array),len(land_array[0]),4))
    nodata_value = arcpy.Raster(land_map).noDataValue
    focuszone = [500,1000,500,1000]
#    focuszone = [0,len(land_array)]
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
    ax4.set_title('Risks')
##    ax5 = plt.subplot2grid((fig_row,fig_col),(map_row+1,0),colspan=fig_col/2,rowspan=1)
##    ax5.set_xlabel('Ecologic risk')
##    ax6 = plt.subplot2grid((fig_row,fig_col),(map_row+1,fig_col/2),colspan=fig_col/2,rowspan=1)
##    ax6.set_xlabel('Agricultural risk')
    index = 0
    for t in type_list:
        plt.subplot2grid((fig_row,fig_col),(map_row+max([map_row/2,1]),0+2*index),colspan=2,rowspan=max([map_row/2,1]))
        plt.title('Area of {0}'.format(type_name[t]))
        plt.subplot2grid((fig_row,fig_col),(map_row+2*max([map_row/2,1]),0+2*index),colspan=2,rowspan=max([map_row/2,1]))
        plt.title('Shape index of {0}'.format(type_name[t]))
        index += 1  
    plt.tight_layout()
    plt.pause(0.01)
## display land use map
    ####*** start optimization iteration  
    update_interval = (focuszone[1]-focuszone[0])*2
    max_iteration = update_interval * 10000
    window_size = 3
## using the random array for template
    value_maps = {'SOCP_1':np.random.random([focuszone[1]-focuszone[0],focuszone[3]-focuszone[2]])}
    risk_maps = {'SC_1':np.random.random([focuszone[1]-focuszone[0],focuszone[3]-focuszone[2]]),
                 'SC_2':np.random.random([focuszone[1]-focuszone[0],focuszone[3]-focuszone[2]])}
  
    area_list_show,SA_list_show,value_list_show,risk_list_show = [],[],[],[]

    k = 0
    while True:
        i,j = random.randint(focuszone[0],focuszone[1]),random.randint(focuszone[2],focuszone[3])
##        for i in range(focuszone[0],focuszone[1]):
##            for j in range(focuszone[2],focuszone[3]):
        if land_array[i,j] != nodata_value:
            or_type,fi_type = land_array[i,j],land_array[i,j]
            neighbor = land_array[max([0,i-window_size]):min([i+window_size,len(land_array)]),
                                  max([0,j-window_size]):min([j+window_size,len(land_array[0])])]
            for t in list(np.unique(neighbor)):
                if t != or_type and t!= nodata_value:
                    if np.sum(neighbor==t) > np.sum(neighbor==fi_type):
                        fi_type = t
        ## upgrade the type and axis
            if or_type != fi_type:
                print "location ({0},{1})change from {2} to {3}".format(i,j,or_type,fi_type)               
                land_array[i,j]= fi_type
                land_array_show[i,j] = cmap[fi_type][:3]+[1.0]                                  
            if k % update_interval == 0:
                ax2.cla()
                ax2.imshow(land_array_show[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],interpolation='nearest')
                ax2.set_title('Optimized land use map (iteration={0})'.format(k))
### calculate and draw value maps
                value_list = values_calculate(land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],value_maps,nodata_value)
                value_list_show.append(value_list)
                ax3.cla()
                ax3.set_title('Values')
                ax3.get_yaxis().get_major_formatter().set_useOffset(False)
                for value in value_list.keys():
                    ax3.plot([x[value] for x in value_list_show],label = value)
                ax3.legend()
## calculate and draw risk maps
                risk_list = values_calculate(land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],risk_maps,nodata_value)
                risk_list_show.append(risk_list)
                ax4.cla()
                ax4.set_title('Risks')
                ax4.get_yaxis().get_major_formatter().set_useOffset(False)
                for risk in risk_list.keys():
                    ax4.plot([x[risk] for x in risk_list_show],label = risk)
                ax4.legend()
## calculate and draw area and shape index changes
                index = 0
                area_list_show.append(area_sta(land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],type_list))                    
                SA_list_show.append(shape_index(land_array[focuszone[0]:focuszone[1],focuszone[2]:focuszone[3]],type_list,nodata_value,window_size))
                for t in type_list:
                    sub_ax1 = plt.subplot2grid((fig_row,fig_col),(map_row+max([map_row/2,1]),0+2*index),colspan=2,rowspan=max([map_row/2,1]))
                    sub_ax1.cla()
                    sub_ax1.get_yaxis().get_major_formatter().set_useOffset(False)
                    sub_ax1.set_title('Area of {0}'.format(type_name[t]))
##                        plt.xlim(0,max_iteration/update_interval+1)
                    sub_ax1.plot([x[t] for x in area_list_show],'-',label=type_name[t],color = cmap[t])
                    sub_ax2 = plt.subplot2grid((fig_row,fig_col),(map_row+2*max([map_row/2,1]),0+2*index),colspan=2,rowspan=max([map_row/2,1]))
                    sub_ax2.cla()
                    sub_ax2.set_title('Shape index of {0}'.format(type_name[t]))
                    sub_ax2.get_yaxis().get_major_formatter().set_useOffset(False)
                    sub_ax2.plot([x[t] for x in SA_list_show],'-',label=type_name[t],color = cmap[t])
                    index += 1
## update drawing
                plt.pause(0.001)
                if k > max_iteration:
                    break
            k += 1
##    plt.show(block=True)
            
            
    

    
