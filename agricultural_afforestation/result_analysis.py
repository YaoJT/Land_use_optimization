
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from tools import *
    import tools
    weights = [0.5,1.0,2.0,3.0,4.0]
    records = range(6)
    FDR = []
    file = './zoning_results/'
    out_res = np.ones((len(weights),len(records)))
    
    for w in range(len(weights)):
        for r in range(len(records)):
            data = open(file+'weights_{0}_record_{1}.txt'.format(weights[w],records[r])).readlines()[11]
            data = data[7:-2].replace("'",'').replace('{','').replace(' ','').split(',')
            res = {}
            for da in data:
                res[da.split(':')[0]] = float(da.split(':')[1])
            out_res[w,r] = res['Carbonsequestration']
            if w == 0:
                FDR.append(1-res['Foodproductive'])

##    plt.subplot(1,2,1)
##    for i in range(out_res.shape[0]):
##        plt.plot(FDR,out_res[i],label = 'weights = {0}'.format(weights[i]))
##    plt.xticks(FDR)
##    plt.xlabel('Food decrease rate (FDR)')
##    plt.legend()
##    plt.subplot(1,2,2)
    for i in range(out_res.shape[1]):
        x0,y0 = weights[2],out_res.transpose()[i][2]
        x1,y1 = x0+0.5,y0+0.01
        plt.plot(weights,out_res.transpose()[i])
        plt.text(x1,y1,'FDR = {0}'.format(round(FDR[i],2)))
        plt.annotate('',xy=(x0,y0),xytext=(x1,y1),
                     arrowprops=dict(facecolor='black', shrink=-0.05,
                                     width = 0.5, headwidth = 5))
    plt.xticks(weights)
    plt.ylim(out_res.min()-0.05,out_res.max()+0.05)
    plt.xlabel('Spatial compactness weight')
    plt.ylabel('Carbon sequestration potential')
    plt.savefig(file+'quantity_result.png')
    plt.show()
#### drawing results maps
## Optimized zoning of agricultural afforestation with different
##    spatial compactness weights (Food decrease rate = 0.1)
##    from importlib import reload
##    reload(tools)
##    from tools import plotRaster_class

    
## Optimized zoning of agricultural afforestation with different
##    spatial compactness weights (Food decrease rate = 0.1)
    spatialRef = 'data/lucc2015.tif'
    record = 1
    tlabels = ['A','B','C','D','E']
    titles = ['{0}: Spatial compactness weight = {1}'.format(tlabels[i],weights[i])
              for i in range(len(weights))]
    i = 0
    for w in weights:
        asciifile = file+'weights_{0}_record_{1}_data.txt'.format(w,record)
        ff = asciiToRaster(asciifile,asciifile.replace('.txt','.tif'),spatialRef)
        data = toWgs(ff,ff.replace('.tif','_wgs.tif'))
        lucc = gdal.Open(data)
        arrayLucc = lucc.ReadAsArray()
        Geo = lucc.GetGeoTransform()
        minx,maxx = Geo[0],Geo[0]+Geo[1]*arrayLucc.shape[1]
        miny,maxy = Geo[3]+Geo[5]*arrayLucc.shape[0],Geo[3]
        
        subExtent = (116.9, 117.15, 40.5, 40.7)
    ##    plotRaster_class(data,{1:'Agricultural area',2:'afforestation area'})
        
        aa = plotRaster_class(data,{1:'Agricultural area',2:'Afforestation area'},
                         subplot=dict(extent=subExtent,scale=3,loc=4),
                              legend_loc = 'upper left',title = titles[i])
        i += 1

        
#### Optimized zoning of agricultural afforestation with different
####    food decrease rate (Spatial compactness weight = 1)
##    spatialRef = 'data/lucc2015.tif'
##    weight = 1.0
##    tlabels = ['A','B','C','D']
##    FDRs = np.arange(0.05,0.21,0.05)
##    titles = ['{0}: Food decrease rate = {1}'.format(tlabels[i],round(FDRs[i],2))
##              for i in range(len(tlabels))]
##    i = 0
##    for r in records:
##        asciifile = file+'weights_{0}_record_{1}_data.txt'.format(weight,r)
##        ff = asciiToRaster(asciifile,asciifile.replace('.txt','.tif'),spatialRef)
##        data = toWgs(ff,ff.replace('.tif','_wgs.tif'))
##        lucc = gdal.Open(data)
##        arrayLucc = lucc.ReadAsArray()
##        Geo = lucc.GetGeoTransform()
##        minx,maxx = Geo[0],Geo[0]+Geo[1]*arrayLucc.shape[1]
##        miny,maxy = Geo[3]+Geo[5]*arrayLucc.shape[0],Geo[3]
##        
##        subExtent = (116.9, 117.15, 40.5, 40.7)
##    ##    plotRaster_class(data,{1:'Agricultural area',2:'afforestation area'})
##        
##        aa = plotRaster_class(data,{1:'Agricultural area',2:'Afforestation area'},
##                         subplot=dict(extent=subExtent,scale=3,loc=4),
##                              legend_loc = 'upper left',title = titles[i],show=True)
##        i += 1        
##    
            
            
