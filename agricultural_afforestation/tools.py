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
cmap = cm.GMT_split

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

def plotRaster(raster_name,arcgisService='ESRI_Imagery_World_2D',**kwargs):
    """
    arcgisService: Name of arcgisService (http://server.arcgisonline.com/arcgis/rest/services)
    or Flase
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
##    plt.title('food productivity of Beijing, 2015')
    out_file = kwargs['out_file'] if 'out_file' in kwargs.keys() else raster_name.replace('.tif','.png')
    plt.savefig(out_file)
    print(out_file)
    show = False if 'show' not in kwargs.keys() else kwargs['show']
    if show:
        plt.show()
    else:
        plt.close()
