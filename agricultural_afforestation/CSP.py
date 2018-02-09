## estimating the carbon sequestration potential of 1km *1km grids in Beijing

    
    
if __name__ == '__main__':
    from tools import *
    import math
    import tools
    from shutil import copyfile
    tools.cmap = cm.GMT_haxby
    spatialRef = './data/lucc2015.tif'
    arraylucc,info = asciiToArray('./data/100m/lucc2015.txt','int',True)
    arraylucc = (arraylucc==1)
    
    arrayDem = asciiToArray('./data/100m/dem.txt','float')
    arrayDem = arrayDem*(arrayDem>=1)+(arrayDem<1)
    arrayMat = asciiToArray('./data/100m/mat.txt','float')
    arrayMat = arrayMat*(arrayMat>=1)+(arrayMat<1)
    arrayMap = asciiToArray('./data/100m/map.txt','float')
    arrayMap = arrayMap*(arrayMap>=1)+(arrayMap<1)
    

#### carbon sequestration potential of SOC
#### pre-processing of data
##    soil_P = pd.read_csv('./data/soil_properties.csv')
##    soil_P = soil_P.set_index('Value')
##    for i in soil_P.index:
##        soil_P['T_OC'][i] = soil_P['T_OC'][i] if soil_P['T_OC'][i] < 10 else soil_P['T_OC'][i]/100
##        soil_P['S_OC'][i] = soil_P['S_OC'][i] if soil_P['S_OC'][i] < 10 else soil_P['S_OC'][i]/100
##    columns = ['T_OC','T_CLAY','T_REF_BULK_DENSITY','T_GRAVEL','T_PH_H2O',
##               'S_OC','S_CLAY','S_REF_BULK_DENSITY','S_GRAVEL','S_PH_H2O',]
##    for c in columns:
##        xx = [x for x in soil_P.index if soil_P[c][x] == 0]
##        soil_P[c][xx] = soil_P[c].mean()
##    soil_P.to_csv('./data/soil_properties_modified.csv')
## start from here if don't need to modify
    
    soil_P = pd.read_csv('./data/soil_properties_modified.csv')
    soil_P.set_index('Value')
##    soil_P['T_SOC'] = soil_P['T_OC']*soil_P['T_REF_BULK_DENSITY']*(1-soil_P['T_GRAVEL']/100)*30
    soil_P['T_SOC'] = soil_P['T_OC']*soil_P['T_REF_BULK_DENSITY']*(1-soil_P['T_GRAVEL']/100)*20
##    soil_P['S_SOC'] = soil_P['S_OC']*soil_P['S_REF_BULK_DENSITY']*(1-soil_P['S_GRAVEL']/100)*70
##    soil_P['SOC100'] = soil_P['T_SOC'] + soil_P['S_SOC']

      
##  current SOC
    arraySoil = gdal.Open('./data/beijing_soil_code.tif').ReadAsArray()
    arraySoil = arraySoil[2:arraylucc.shape[0]+2,2:arraylucc.shape[1]+2]
    arraySOC = np.array([[soil_P['T_SOC'][x] if x in soil_P.index else soil_P['T_SOC'].mean()
                         for x in y] for y in arraySoil])
##    arraySOC = np.array([[soil_P['SOC100'][x] if x in soil_P.index else soil_P['SOC100'].mean()
##                         for x in y] for y in arraySoil])
    
    arraySOC = arraySOC*arraylucc
    SOC_1000 = dissolve_sum(arraySOC/1000,info,'./data/SOC_1km.txt')
    asciiToRaster('./data/SOC_1km.txt','./data/Raster/SOC_1km.tif',spatialRef)
    toWgs('./data/Raster/SOC_1km.tif','./data/Raster/SOC_1km_wgs.tif')
    plotRaster('./data/Raster/SOC_1km_wgs.tif',False)
## SOC saturation estimated based on the SOC saturation knowledge, 0-20 cm
    arraySOCS = arraySOC.copy()
    for i in range(arraySOCS.shape[0]):
        for j in range(arraySOCS.shape[1]):
            if arraySOCS[i,j] >0:
                MT,MW = arrayMat[i,j],arrayMap[i,j]/100
                CL = soil_P['T_CLAY'][arraySoil[i,j]] if arraySoil[i,j] in soil_P.index else soil_P['T_CLAY'].mean()
                PH = soil_P['T_PH_H2O'][arraySoil[i,j]] if arraySoil[i,j] in soil_P.index else soil_P['T_PH_H2O'].mean()
                arraySOCS[i,j] = 140.5*math.pow(math.e,-0.021*MT)-98.8*math.pow(math.e,-0.42*MW)-39.6*math.pow(math.e,-0.1*CL)-4.1*PH-27.7
##            arraySOCS[i,j] = arraySOC[i,j] if arraySOCS[i,j] < arraySOC[i,j] else arraySOCS[i,j]
    SOCS_1000 = dissolve_sum(arraySOCS/1000,info,'./data/SOCS_1km.txt')
    asciiToRaster('./data/SOCS_1km.txt','./data/Raster/SOCS_1km.tif',spatialRef)
    toWgs('./data/Raster/SOCS_1km.tif','./data/Raster/SOCS_1km_wgs.tif')
    plotRaster('./data/Raster/SOCS_1km_wgs.tif',False)
    
                
        
    
## SOC after afforestation estimated based on the SOC of forest 0-100cm
##    arraySOCF = np.power(10,((0.340+0.347*np.log10(arrayDem)-0.355*np.log10(arrayMat)+0.360*np.log10(arrayMap))))
##    arraySOCF = arraySOCF*(arraySOCF>arraySOC)+arraySOC
##    arraySOCF = arraySOCF.data*arraylucc
##    SOCF_1000 = dissolve_sum(arraySOCF/1000,info,'./data/SOCF_1km.txt')
##    asciiToRaster('./data/SOCF_1km.txt','./data/Raster/SOCF_1km.tif',spatialRef)
##    toWgs('./data/Raster/SOCF_1km.tif','./data/Raster/SOCF_1km_wgs.tif')
##    plotRaster('./data/Raster/SOCF_1km_wgs.tif',False)


## SOCP plotting
##    arraySOCP = arraySOCF - arraySOC
    arraySOCP = arraySOCS - arraySOC
    SOCP_1000 = dissolve_sum(arraySOCP/1000,info,'./data/SOCP_1km.txt')
    asciiToRaster('./data/SOCP_1km.txt','./data/Raster/SOCP_1km.tif',spatialRef)
    toWgs('./data/Raster/SOCP_1km.tif','./data/Raster/SOCP_1km_wgs.tif')
    plotRaster('./data/Raster/SOCP_1km_wgs.tif',False)

## carbon sequestration sequestration of plant biomass
    arrayCPV = (74.52+0.0352*arrayMap+3.5474*arrayMat)
    arrayCPV = arrayCPV.data*arraylucc
    CPV_1000 = dissolve_sum(arrayCPV/1000,info,'./data/CPV_1km.txt')
    asciiToRaster('./data/CPV_1km.txt','./data/Raster/CPV_1km.tif',spatialRef)
    toWgs('./data/Raster/CPV_1km.tif','./data/Raster/CPV_1km_wgs.tif')
    plotRaster('./data/Raster/CPV_1km_wgs.tif',False)

## carbon sequestration sequestration of agricultural afforestation
    arrayCP = arrayCPV+arraySOCP
    CP_1000 = dissolve_sum(arrayCP/1000,info,'./data/CP_1km.txt')
    copyfile('./data/CP_1km.txt','./data/1000m/CP.txt')
    asciiToRaster('./data/CP_1km.txt','./data/Raster/CP_1km.tif',spatialRef)
    toWgs('./data/Raster/CP_1km.tif','./data/Raster/CP_1km_wgs.tif')
    plotRaster('./data/Raster/CP_1km_wgs.tif',False)
    
    ax = plt.subplot(1,1,1)
    ax.boxplot([[x for x in SOC_1000.flatten() if x > 0],
                 [x for x in SOCS_1000.flatten() if x >0],
                 [x for x in SOCP_1000.flatten() if x >0],
                 [x for x in CPV_1000.flatten() if x > 0],
                 [x for x in CP_1000.flatten() if x > 0]])
    ax.set_xticklabels(['SOC','SOCS','CPS','CPV','CP'])
    plt.savefig('./data/Raster/CP_box.png')
    plt.show()


    
    
    
    
    





















    
