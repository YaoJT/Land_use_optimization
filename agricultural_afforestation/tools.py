import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import gdal
import osr
import math
import pyodbc
import sys
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.patches as mpatches
cmap = cm.GMT_relief_r
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 14

def asciiToArray(file_path,arrayType='int',return_detail=False):
    array = open(file_path).readlines()[6:]
    if arrayType == 'int':
        array = [[int(x) for x in y.split()] for y in array]
    elif arrayType == 'float':
        array = [[float(x) for x in y.split()] for y in array]
    if return_detail:
        detail = open(file_path).readlines()[:6]
        detail_dic = {}
        for d in detail:
            detail_dic[d.split()[0]] = float(d.split()[1])
        return np.array(array),detail_dic
    else:
        return np.array(array)
def arrayToAscii(in_array,detail,out_path):
    out_file = open(out_path,'w+')
    for x in detail:
        out_file.write('{0}     {1}\n'.format(x,detail[x]))
    for x in in_array:
        for y in x:
            out_file.write(str(y).replace('--',str(detail['NODATA_value']))+'    ')
        out_file.write('\n')
    out_file.close()
    print(out_file)
def asciiToRaster(ascii_file,out_file,spatialRef):
    array,info = asciiToArray(ascii_file,'float',True)

    pixelWidth = info['cellsize']
    pixelHeight = -info['cellsize']
    cols = info['ncols']
    rows = info['nrows']
    originX = info['xllcorner']
    originY = info['yllcorner']-pixelHeight*rows
    if os.path.exists(os.path.split(out_file)[0])==False:
        os.makedirs(os.path.split(out_file)[0])
    
    
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(out_file,int(cols),int(rows),1, gdal.GDT_Float64)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outRaster.GetRasterBand(1).WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    try:
        outRasterSRS.ImportFromWkt(gdal.Open(spatialRef).GetProjectionRef())
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
    except:
        outRaster.SetProjection(spatialRef)
    outRaster.FlushCache()
    print(out_file)
    return out_file
def dissolve_sum(in_array,info,out_file,scale=10):
    if os.path.exists(os.path.split(out_file)[0]) == False:
        os.makedirs(os.path.split(out_file)[0])
    info=info.copy()
    rows = info['nrows']
    cellsize = info['cellsize']
    info['cellsize'] = info['cellsize']*scale
    info['nrows'] = info['nrows']//scale if info['nrows']%scale == 0 else info['nrows']//scale+1
    info['ncols'] = info['ncols']//scale if info['ncols']%scale == 0 else info['ncols']//scale+1
    info['yllcorner'] = info['yllcorner'] - cellsize*(info['nrows']*scale-rows)
    out_array = np.zeros((int(info['nrows']),int(info['ncols'])))
    for i in range(int(info['nrows'])):
        for j in range(int(info['ncols'])):
            out_array[i,j] = np.sum(in_array[i*scale:min(i*scale+scale,in_array.shape[0]),
                                          j*scale:min(j*scale+scale,in_array.shape[1])])
            
    arrayToAscii(out_array,info,out_file)
    print(out_file)
    return out_array
    
def toWgs(in_raster,out_raster):
    os.system('gdalwarp {0} {1} -t_srs "+proj=longlat +ellps=WGS84"'.format(in_raster,out_raster))
    return out_raster

def plotRaster(raster_name,arcgisService='ESRI_Imagery_World_2D',**kwargs):
    """
    arcgisService: Name of arcgisService (http://server.arcgisonline.com/arcgis/rest/services)
    or Flase
    kwargs:
    subplot (dict): insert a subplot in the fig
        extent: x0,x1,y0,y1, extent of the subplot
        scale: float, scale of the subplot
    """
    FP = gdal.Open(raster_name)
    data = FP.ReadAsArray()
    minx,maxy = FP.GetGeoTransform()[0],FP.GetGeoTransform()[3]
    xsize,ysize=FP.GetGeoTransform()[1],FP.GetGeoTransform()[5]
    maxx,miny = minx+xsize*FP.RasterXSize,maxy+ysize*FP.RasterYSize
    loncorners = [minx,maxx]
    latcorners = [miny,maxy]
    # create figure and axes instances
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    # create polar stereographic Basemap instance.
    epsg = 3395 if 'epsg' not in kwargs.keys() else kwargs['epsg']

    m = Basemap(epsg=epsg,
                llcrnrlat=miny,llcrnrlon=minx,urcrnrlat=maxy,
                urcrnrlon=maxx)
    if arcgisService != False:
        m.arcgisimage(service=arcgisService, xpixels =2000,verbose= True)
    # draw paralle
    parallels = np.arange(39,42,0.5)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    # draw meridians
    meridians = np.arange(115,118.5,0.5)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    ny = data.shape[0]; nx = data.shape[1]
    lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
    lons,lats = np.meshgrid(np.arange(minx,maxx,xsize),
                            np.arange(maxy,miny,ysize))
    x, y = m(lons, lats) # compute map proj coordinates.
    # draw filled contours.
    clevs = list(range(0,1000,100))
    data = np.ma.masked_less_equal(data,0)
    cs = m.pcolormesh(x,y,data,cmap=cmap)
    # draw province and counties from shapefiles.
##    m.readshapefile('./data/province','province',linewidth=1.5)
    try:
        m.readshapefile(kwargs[base_shp],'baseshp',linewidth=1)
    except:
        m.readshapefile('./data/Beijing_county','baseshp',linewidth=1)
    # add colorbar.
    cbar = m.colorbar(cs,location='bottom',pad="5%")
    label = kwargs['label'] if 'label' in kwargs.keys() else ''
    cbar.set_label(label)
    # add title
    if 'title' in kwargs.keys():
        plt.title(kwargs['title'])
    out_file = kwargs['out_file'] if 'out_file' in kwargs.keys() else raster_name.replace('.tif','.png')
    plt.savefig(out_file)
    print(out_file)
    show = False if 'show' not in kwargs.keys() else kwargs['show']
    if 'subplot' in kwargs.keys():
        axins = zoomed_inset_axes(ax, kwargs['subplot']['scale'], loc='best')
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        extent = kwargs['subplot']['extent']
        map2 = Basemap(llcrnrlon=extent[0],llcrnrlat=extent[2],
                       urcrnrlon=extent[1],urcrnrlat=extent[3],ax=axins)
        map2.pcolormesh(x,y,data,cmap=cmap)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    if show:
        plt.show()
    else:
        plt.close()

def plotRaster_class(inRaster,labels,arcgisService = False,**kwargs):
    lucc = gdal.Open(inRaster)
    arrayLucc = lucc.ReadAsArray()
    Geo = lucc.GetGeoTransform()
    minx,maxx = Geo[0],Geo[0]+Geo[1]*arrayLucc.shape[1]
    miny,maxy = Geo[3]+Geo[5]*arrayLucc.shape[0],Geo[3]
    t = 200
    if 'cmap' not in kwargs.keys():
        cmap = {1:[244,241,66,t],2:[58,193,96,t],3:[171,216,117,t],
                4:[71,141,232,t],5:[168,5,21,t],6:[255,255,255,t],
                'NA':[255,255,255,0]}
    else:
        cmap = kwargs['cmap']
    for x in cmap:
        cmap[x] = [c/255.0 for c in cmap[x]]
##    labels = {1:'Agricultural land',2:'Forest land',3:'Grassland',
##              4:'Water area',5:'Built-up land'}

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.1,0.05,0.85,0.95])
    try:
        plt.title(kwargs['title'])
    except:
        None

    plt.xticks(visible=False)
    plt.yticks(visible=False)
    

    map = Basemap(epsg=3395, llcrnrlon=minx,llcrnrlat=miny,
                   urcrnrlon=maxx,urcrnrlat=maxy)
    map.readshapefile('./data/Beijing_county','Beijing')
##    map.readshapefile('./data/province','province',linewidth=1)
    
    x = np.linspace(minx,maxx,arrayLucc.shape[1])
    y = np.linspace(miny,maxy,arrayLucc.shape[0])
    map.drawmeridians(np.arange(int(minx),int(maxx)+1,0.5),labels=[0,0,0,1],
                       linewidth=1)
    map.drawparallels(np.arange(int(miny),int(maxy)+1,0.5),labels=[1,0,0,0],
                       linewidth=1)
    if arcgisService != False:
        map.arcgisimage(service=arcgisService, xpixels = 800,
                         verbose= True)
    xx,yy = np.meshgrid(x,y)
    arrayLucc = np.array([[x if x <10 else x//10 for x in y] for y in arrayLucc])
    arrayShow = np.array([[cmap[x] if x in labels.keys() else cmap['NA'] for x in y]
                          for y in arrayLucc])
    mx,my=map(maxx,maxy)
    map.imshow(np.flipud(arrayShow))
    patches = [mpatches.Patch(color=cmap[x],label=labels[x]) for x in labels]
    legend_loc = 'best' if 'legend_loc' not in kwargs.keys() else kwargs['legend_loc']
    plt.legend(handles = patches,loc=legend_loc, 
               frameon=True,edgecolor='0.0',title='Legend')
    if 'tilte' in kwargs.keys():
        plt.title(kwargs['title'])
    if 'subplot' in kwargs.keys():
## draw rectanle in the ax plot
        axins = zoomed_inset_axes(ax, kwargs['subplot']['scale'], loc=kwargs['subplot']['loc'])
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        extent = kwargs['subplot']['extent']
        x0,x1 = int((extent[0]-minx)/Geo[1]),int((extent[1]-minx)/Geo[1])
        y0,y1 = -int((maxy-extent[3])/Geo[5]),-int((maxy-extent[2])/Geo[5])
##        print(x0,y0,x1,y1,miny,maxy)
        map2 = Basemap(epsg = 3395, llcrnrlon=extent[0],llcrnrlat=extent[2],
                       urcrnrlon=extent[1],urcrnrlat=extent[3],ax = axins)
        map2.imshow(np.flipud(arrayShow[y0:y1,x0:x1]))
## draw rectanle in the ax plot
        x,y = map(extent[0],extent[2])
        xx,yy = map(extent[1],extent[3])
        height,width = xx-x,yy-y
        ax.add_patch(mpatches.Rectangle((x,y),height,width,ec='0.0',fc='none',lw=1.5))
##        aa = mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")       
##    plt.tight_layout()
    plt.savefig(inRaster.replace('.tif','.png'))
    show = False if 'show' not in kwargs.keys() else kwargs['show']
    if show:
        plt.show()
