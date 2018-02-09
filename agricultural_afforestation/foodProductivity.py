## estimating the food productivity of 1km * 1km grids in Beijing

    
    
    

if __name__ == '__main__':
    from tools import *
    from shutil import copyfile
    import tools
    tools.cmap = cm.GMT_haxby
##    arrayXzq = asciiToArray('./data/100m/xzq.txt')
##    FCR = pd.DataFrame.from_csv('./data/FCR.csv')
####    FCR['FCR']=FCR['FCA']/FCR['Area_map']
##    for x in np.unique(arrayXzq):
##        if x not in FCR.index:
##            FCR.loc[x] = 0
##    arrayFCR = np.array([[FCR['FCR'][x] for x in y] for y in arrayXzq])
##    arrayP = asciiToArray('./data/100m/pcrop_rainfed_maize.txt','float')
##    arraylucc,info = asciiToArray('./data/100m/lucc2015.txt','int',True)
##    newRow = int(info['nrows']/10)+1 if info['nrows']%10 !=0 else int(info['nrows']/10)
##    newCol = int(info['ncols']/10)+1 if info['ncols']%10 !=0 else int(info['ncols']/10)
##    FPArray = np.zeros((newRow,newCol))
##    FPArray += info['NODATA_value']
##
##    arraylucc=arraylucc[:2024]
##    arrayP=arrayP[:2024]
##    for i in FCR.index:
##        FCR['FP_GAEZ'][i] = np.sum((arraylucc==1)*(arrayXzq==i)*arrayP*FCR['FCR'][i])
##    FCR.to_csv('./data/FCR.csv')
#### validating of GAEZ and ploting
##    FCR = pd.read_csv('./data/FCR.csv')
##    df = FCR.loc[[x for x in FCR.index if FCR['Area_map'][x]>0]][['County','FP_real','FP_GAEZ']]
##    x,y = df['FP_real']/10000,df['FP_GAEZ']/10000
##    plt.scatter(x,y,c='0.3',marker='x')
##    z1 = np.polyfit(x,y,1)
##    p1 = np.poly1d(z1)
##    plt.plot([0,x.max()],p1([0,x.max()]),'--',c='0')
##    plt.text(x.max()/2-2,y.max()-1,'y = {0}*x+{1} '.format(round(z1[0],4),round(z1[1],4)))
##    plt.text(x.max()/2-2,y.max()-2,'$R^{2}=$'+str(df['FP_real'].corr(df['FP_GAEZ'])**2)[:6])
##    plt.xlabel('Actual yield (ten thousands tons)')
##    plt.ylabel('Calculated yield with GAEZ (ten thousands tons)')
##    plt.savefig('./data/GAEZ_validation.png')
##    plt.show()
    
    
####  dissolve to 1km grids， calculating from the left-bottom corner
##    for r in range(newRow):
##        for c in range(newCol):
##            maxrow = min(r*10+10,len(arrayP),len(arrayFCR),len(arraylucc))
##            maxcol = min(c*10+10,len(arrayP[0]),len(arrayFCR[0]),len(arraylucc[0]))            
##            Pn = arrayP[r*10:maxrow,c*10:maxcol]
##            FCRn = arrayFCR[r*10:maxrow,c*10:maxcol]
##            luccn = arraylucc[r*10:maxrow,c*10:maxcol]
##            FPArray[r,c] = np.sum(Pn*FCRn*(luccn == 1)) if np.sum(luccn==1) > 0 else info['NODATA_value']
##    newInfo = info.copy()
##    newInfo['nrows'] = newRow
##    newInfo['ncols'] = newCol
##    newInfo['cellsize'] = 1000
##    newInfo['yllcorner'] += info['cellsize']*(info['nrows']-newRow*10)
##    arrayToAscii(FPArray,newInfo,'./data/foodProductivity.txt')
## draw maps
    copyfile('./data/foodProductivity.txt','./data/1000m/FP.txt')
##    cmap1 = LinearSegmentedColormap.from_list("my_colormap", ((0, 1, 0), (1, 0, 0)), N=6, gamma=1.0)   
    FR = asciiToRaster('./data/1000m/FP.txt','./data/Raster/FP_1km.tif','./data/100m/lucc2015.tif')
    os.system('gdalwarp {0} {1} -t_srs "+proj=longlat +ellps=WGS84"'.format('./data/Raster/FP_1km.tif','./data/Raster/FP_1km_wgs.tif'))
    plotRaster('./data/Raster/FP_1km_wgs.tif',False)

##    
##    FP = gdal.Open('./data/Raster/FP_wgs.tif')
##    
##    data = FP.ReadAsArray()
##    minx,maxy = FP.GetGeoTransform()[0],FP.GetGeoTransform()[3]
##    xsize,ysize=FP.GetGeoTransform()[1],FP.GetGeoTransform()[5]
##    maxx,miny = minx+xsize*FP.RasterXSize,maxy+ysize*FP.RasterYSize
##    loncorners = [minx,maxx]
##    latcorners = [miny,maxy]
##    # create figure and axes instances
##    fig = plt.figure(figsize=(8,8))
##    ax = fig.add_axes([0.1,0.1,0.8,0.8])
##    # create polar stereographic Basemap instance.
##
##    m = Basemap(epsg=3395,llcrnrlat=miny,
##                llcrnrlon=minx,urcrnrlat=maxy,
##                urcrnrlon=maxx)
##    # draw paralle
##    parallels = np.arange(39,42,0.5)
##    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
##    # draw meridians
##    meridians = np.arange(115,118.5,0.5)
##    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
##    x = np.linspace(minx,maxx, data.shape[1])
##    y = np.linspace(miny,maxy, data.shape[0])
##    xx,yy = np.meshgrid(x,y)
##
##    data = np.ma.masked_less_equal(data,0)
##
####    m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels =2000,verbose= True)
####    cs = m.pcolormesh(xx,yy,data,cmap=cm.s3pcpn,latlon=True)
##    cs = m.pcolormesh(xx,yy,np.flipud(data),latlon=True,cmap=cm.s3pcpn_l_r)
##    # draw province and counties from shapefiles.
####    m.readshapefile('./data/province','province',linewidth=1.5)
##    m.readshapefile('./data/Beijing_county','Beijing',linewidth=1)
##    # add colorbar.
##    cbar = m.colorbar(cs,location='bottom',pad="5%")
####    cbar.set_label('ton')
##    # add title
####    plt.title('food productivity of Beijing, 2015')
##    plt.savefig('food_productivity.png')
##    plt.show()
##            
    
    
    
    
