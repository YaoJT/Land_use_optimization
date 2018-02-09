## draw lucc map of Beijing 2015

if __name__ == '__main__':
    from tools import *
    import matplotlib
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    import matplotlib.patches as mpatches
    lucc = gdal.Open('./data/100m/lucc2015_wgs.tif')
    arrayLucc = lucc.ReadAsArray()
    Geo = lucc.GetGeoTransform()
    minx,maxx = Geo[0],Geo[0]+Geo[1]*arrayLucc.shape[1]
    miny,maxy = Geo[3]+Geo[5]*arrayLucc.shape[0],Geo[3]
    minx_c,maxx_c = 72.5, 135.5
    miny_c,maxy_c = 4,53.5
    minx_h,maxx_h = 113,120
    miny_h,maxy_h = 36,42.5
    t = 200
    cmap = {1:[244,241,66,t],2:[58,193,96,t],3:[171,216,117,t],
            4:[71,141,232,t],5:[168,5,21,t],6:[255,255,255,t],
            'NA':[255,255,255,0]}
    for x in cmap:
        cmap[x] = [c/255.0 for c in cmap[x]]
    labels = {1:'Agricultural land',2:'Forest land',3:'Grassland',
              4:'Water area',5:'Built-up land'}

    fig = plt.figure(figsize=(4,5))
##    ax = fig.add_subplot(111)
#### world
##    m = Basemap(lon_0 = minx_c+40)
##    m.drawcountries()
##    m.drawcoastlines()
##    m.drawmeridians(range(0,360,20),labels=[1,1,1,1])
##    m.drawparallels(range(-90,90,10),labels=[1,1,1,1])
####    m.shadedrelief()
##    
##
##    ax1 = zoomed_inset_axes(ax, 2, loc=3)
####    ax1.set_xlim(minx-30,maxx+30)
####    ax1.set_ylim(miny-20,maxy+20)    
###### China    
##    map = Basemap(projection='cyl',
##                  llcrnrlon=minx_c,llcrnrlat=miny_c,urcrnrlon=maxx_c,urcrnrlat=maxy_c,
##                  lat_0=35, lon_0=112,ax=ax1)
##    map.readshapefile('./data/province','province')
##    map.drawmapboundary(fill_color='#7777ff')
##    map.drawcountries()
####    map.shadedrelief()
##    map.fillcontinents(color='#ddaa66', lake_color='#7777ff', zorder=0)
##    map.drawcoastlines()
##    mark_inset(ax, ax1, loc1=1, loc2=2, fc="none", ec="0.5")
###### Hebei Province
##    ax2 = zoomed_inset_axes(ax, 10, loc=8)
##    map_p = Basemap(projection='cyl',
##                    llcrnrlon=minx_h,llcrnrlat=miny_h,
##                    urcrnrlon=maxx_h,urcrnrlat=maxy_h,ax=ax2)
##    map_p.readshapefile('./data/province','province')
####    map_p.shadedrelief()
##    mark_inset(ax1, ax2, loc1=2, loc2=3, fc="none", ec="0.5")
##
####    lons = np.array([-13.7, -10.8, -13.2, -96.8, -7.99, 7.5, -17.3, -3.7])
####    lats = np.array([9.6, 6.3, 8.5, 32.7, 12.5, 8.9, 14.7, 40.39])
####    cases = np.array([1971, 7069, 6073, 4, 6, 20, 1, 1])
####    deaths = np.array([1192, 2964, 1250, 1, 5, 8, 0, 0])
####    places = np.array(['Guinea', 'Liberia', 'Sierra Leone','United States', 'Mali', 'Nigeria', 'Senegal', 'Spain'])
####
####    x, y = map(lons, lats)
####
####    map.scatter(x, y, s=cases, c='r', alpha=0.5)
##    
###### draw Beijing
##    axins = zoomed_inset_axes(ax, 50, loc=1)
####    axins.set_xlim(minx,maxx)
####    axins.set_ylim(miny,maxy)

    plt.xticks(visible=False)
    plt.yticks(visible=False)
    

    map2 = Basemap(llcrnrlon=minx,llcrnrlat=miny,
                   urcrnrlon=maxx,urcrnrlat=maxy,epsg=3395)
    map2.readshapefile('./data/Beijing_county','Beijing')
    map2.readshapefile('./data/province','province',linewidth=1)
    
##    map2.shadedrelief()
    x = np.linspace(minx,maxx,arrayLucc.shape[1])
    y = np.linspace(miny,maxy,arrayLucc.shape[0])
    map2.drawmeridians(np.arange(int(minx),int(maxx)+1,0.5),labels=[0,0,0,1],
                       linewidth=1)
    map2.drawparallels(np.arange(int(miny),int(maxy)+1,0.5),labels=[1,0,0,0],
                       linewidth=1)
    map2.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 800,
                     verbose= True)
    xx,yy = np.meshgrid(x,y)
    arrayLucc = np.array([[x if x <10 else x//10 for x in y] for y in arrayLucc])
    arrayShow = np.array([[cmap[x] if x in cmap.keys() else cmap['NA'] for x in y]
                          for y in arrayLucc])
    mx,my=map2(maxx,maxy)
    map2.imshow(np.flipud(arrayShow))
    patches = [mpatches.Patch(color=cmap[x],label=labels[x]) for x in labels]
    plt.legend(handles = patches,loc='lower right', 
               frameon=True,edgecolor='0.0',title='Legend')

####    mark_inset(ax2, axins, loc1=3, loc2=4, fc="none", ec="0.5")
##    plt.savefig('./lucc.png')
    plt.show()
    
    
    
    
