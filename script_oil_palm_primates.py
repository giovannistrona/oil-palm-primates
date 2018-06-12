from copy import deepcopy
from functools import partial
from math import log
from numpy import mean,zeros,array,where,vectorize,uint32
from os import listdir
from osgeo import gdal_array,gdal,gdalconst,ogr
from random import random,sample,shuffle
from rasterio import features
from rasterio.features import shapes
from shapely.geometry import shape,mapping
from shapely.geometry import Polygon as Pol
from shapely.wkb import loads
import fiona
import os
import pyproj    
import rasterio
import shapefile
import shapely.ops as ops
import numpy
from osgeo import osr, ogr, gdal
import numpy as np
import itertools



####for maps
from colormaps import plasma
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap






if not os.path.exists('./various_rasters'):
	os.makedirs('./various_rasters')

if not os.path.exists('./primate_rasters'):
	os.makedirs('./primate_rasters')


if not os.path.exists('./primate_range_reduction_figures'):
	os.makedirs('./primate_range_reduction_figures')


if not os.path.exists('./results'):
	os.makedirs('./results')





# ===============================================
# Filter a shapefile
# ===============================================
# https://joeahand.com/archive/filtering-a-shapefile-with-python/


def create_filtered_shapefile(value, filter_field, in_shapefile,out_shapefile):
	input_ds = ogr.Open(in_shapefile)
	input_layer = input_ds.GetLayer()
	query_str = "{} = '{}'".format(filter_field, value)
	input_layer.SetAttributeFilter(query_str)
	driver = ogr.GetDriverByName("ESRI Shapefile")
	out_ds = driver.CreateDataSource(out_shapefile)
	out_layer = out_ds.CopyLayer(input_layer, str(value))
	del input_layer, out_layer, out_ds
	return out_shapefile



# Filter
create_filtered_shapefile("Africa", "CONTINENT","./continents/continent.shp", "./continents/africa.shp")

driver = ogr.GetDriverByName("ESRI Shapefile")

# input SpatialReference
inSpatialRef = osr.SpatialReference()
inSpatialRef.ImportFromEPSG(4326)

# output SpatialReference
aea_proj = "+proj=aea +lat_1=20 +lat_2=-23 +lat_0=0 +lon_0=25 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m no_defs"
outSpatialRef = osr.SpatialReference()
outSpatialRef.ImportFromProj4(aea_proj)

# create the CoordinateTransformation
coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

# get the input layer
inDataSet = driver.Open("./continents/africa.shp")
inLayer = inDataSet.GetLayer()

# create the output layer
outputShapefile = "./continents/africa_aea.shp"
outDataSet = driver.CreateDataSource(outputShapefile)
outLayer = outDataSet.CreateLayer("africa_aea", srs=outSpatialRef, geom_type=ogr.wkbMultiPolygon)

# add fields
inLayerDefn = inLayer.GetLayerDefn()
for i in range(0, inLayerDefn.GetFieldCount()):
	fieldDefn = inLayerDefn.GetFieldDefn(i)
	outLayer.CreateField(fieldDefn)

# get the output layer's feature definition
outLayerDefn = outLayer.GetLayerDefn()

# loop through the input features
inFeature = inLayer.GetNextFeature()
while inFeature:
	geom = inFeature.GetGeometryRef()
	geom.Transform(coordTrans)
	outFeature = ogr.Feature(outLayerDefn)
	outFeature.SetGeometry(geom)
	for i in range(0, outLayerDefn.GetFieldCount()):
		outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(),inFeature.GetField(i))
	outLayer.CreateFeature(outFeature)
	outFeature = None
	inFeature = inLayer.GetNextFeature()



inDataSet = None
outDataSet = None

# ===============================================
# Make raster
# ===============================================
# https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#convert-an-ogr-file-to-a-raster

# Define pixel_size and NoData value of new raster
pixel_size = 10000
NoData_value = 0

# Filename of input OGR file
vector_fn = "./continents/africa_aea.shp"

# Filename of the raster Tiff that will be created
raster_fn = "./various_rasters/africa.tif"

# Open the data source and read in the extent
source_ds = ogr.Open(vector_fn)
source_layer = source_ds.GetLayer()
x_min, x_max, y_min, y_max = source_layer.GetExtent()

# Create the destination data source
x_res = int(np.ceil((x_max - x_min) / pixel_size))
y_res = int(np.ceil((y_max - y_min) / pixel_size))
target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, x_res, y_res, 1, gdal.GDT_Byte, options=["COMPRESS=LZW"])
target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
target_ds.SetProjection(outSpatialRef.ExportToWkt())
band = target_ds.GetRasterBand(1)
band.SetNoDataValue(NoData_value)

target_ds.GetMetadata()

# Rasterize
gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])
target_ds = None

# ===============================================
# End
# ===============================================


####MISC FUNCTIONS
def round255(x):
	return int(round(round(x*255.0,1)))


def rgb_to_hex(rgb):
	rgb=rgb[:3]
	rgb=map(round255,rgb)
	return '#%02x%02x%02x' % tuple(rgb)





def convert_km(pol):
	return ops.transform(partial(pyproj.transform,pyproj.Proj(init='EPSG:4326'),pyproj.Proj(proj='aea', lat_1=20,lat_2=-23, lat_0=0, lon_0=25, x_0=0, y_0=0,ellps='WGS84',datum='WGS84',units='m',no_defs=True)),pol)



def convert_wgs84(pol):
	return ops.transform(partial(pyproj.transform,pyproj.Proj(proj='aea', lat_1=20,lat_2=-23, lat_0=0, lon_0=25, x_0=0, y_0=0,ellps='WGS84',datum='WGS84',units='m',no_defs=True),pyproj.Proj(init='EPSG:4326')),pol)





referencefile = './various_rasters/africa.tif'#Path to reference file
reference = gdal.Open(referencefile, gdalconst.GA_ReadOnly)
referenceProj = reference.GetProjection()
referenceTrans = reference.GetGeoTransform()
x = reference.RasterXSize 
y = reference.RasterYSize




inputfile = './c_gls_LC100-LCCS_201501010000_AFRI_PROBAV_1.0.1/c_gls_LC100-LCCS_201501010000_AFRI_PROBAV_1.0.1.tiff'
input = gdal.Open(inputfile, gdalconst.GA_ReadOnly)
inputProj = input.GetProjection()
inputTrans = input.GetGeoTransform()


outputfile = './various_rasters/land_use.tif'#Path to output file
driver= gdal.GetDriverByName('GTiff')

bandreference = input.GetRasterBand(1) 
output = driver.Create(outputfile,x,y,1,bandreference.DataType)
output.SetGeoTransform(referenceTrans)
output.SetProjection(referenceProj)

gdal.ReprojectImage(input,output,inputProj,referenceProj,gdalconst.GRA_Mode)

del output


inputfile = './suitability_rasters/rainfed_intermediate/data.asc'
input = gdal.Open(inputfile, gdalconst.GA_ReadOnly)
inputProj = input.GetProjection()
inputTrans = input.GetGeoTransform()


outputfile = './various_rasters/suitability.tif'#Path to output file
driver= gdal.GetDriverByName('GTiff')

bandreference = input.GetRasterBand(1) 
output = driver.Create(outputfile,x,y,1,bandreference.DataType)
output.SetGeoTransform(referenceTrans)
output.SetProjection(referenceProj)

gdal.ReprojectImage(input,output,inputProj,referenceProj,gdalconst.GRA_Mode)

del output




inputfile = './suitability_rasters/rainfed_low/data.asc'
input = gdal.Open(inputfile, gdalconst.GA_ReadOnly)
inputProj = input.GetProjection()
inputTrans = input.GetGeoTransform()


outputfile = './various_rasters/suitability_low.tif'#Path to output file
driver= gdal.GetDriverByName('GTiff')


bandreference = input.GetRasterBand(1) 
output = driver.Create(outputfile,x,y,1,bandreference.DataType)
output.SetGeoTransform(referenceTrans)
output.SetProjection(referenceProj)

gdal.ReprojectImage(input,output,inputProj,referenceProj,gdalconst.GRA_Mode)

del output




inputfile = './suitability_rasters/rainfed_high/data.asc'
input = gdal.Open(inputfile, gdalconst.GA_ReadOnly)
inputProj = input.GetProjection()
inputTrans = input.GetGeoTransform()


outputfile = './various_rasters/suitability_high.tif'#Path to output file
driver= gdal.GetDriverByName('GTiff')

bandreference = input.GetRasterBand(1) 

output = driver.Create(outputfile,x,y,1,bandreference.DataType)
output.SetGeoTransform(referenceTrans)
output.SetProjection(referenceProj)

gdal.ReprojectImage(input,output,inputProj,referenceProj,gdalconst.GRA_Mode)

del output



####CARBON & ACCESSIBILITY

inputfile = './accessibility_carbon/Avitabile_AGB_Map.tif'
input = gdal.Open(inputfile, gdalconst.GA_ReadOnly)
inputProj = input.GetProjection()
inputTrans = input.GetGeoTransform()


outputfile = './various_rasters/carbon.tif'#Path to output file
driver= gdal.GetDriverByName('GTiff')

bandreference = input.GetRasterBand(1) 
output = driver.Create(outputfile,x,y,1,bandreference.DataType)
output.SetGeoTransform(referenceTrans)
output.SetProjection(referenceProj)

gdal.ReprojectImage(input,output,inputProj,referenceProj,gdalconst.GRA_Bilinear)

del output


inputfile = './accessibility_carbon/accessibility_to_cities_2015_v1.0.tif'
input = gdal.Open(inputfile, gdalconst.GA_ReadOnly)
inputProj = input.GetProjection()
inputTrans = input.GetGeoTransform()


outputfile = './various_rasters/accessibility.tif'#Path to output file
driver= gdal.GetDriverByName('GTiff')

bandreference = input.GetRasterBand(1)
bandreference.SetNoDataValue(-9999)
output = driver.Create(outputfile,x,y,1,bandreference.DataType)
output.SetGeoTransform(referenceTrans)
output.SetProjection(referenceProj)

gdal.ReprojectImage(input,output,inputProj,referenceProj,gdalconst.GRA_Bilinear)

del output




####PARKS

rst_fn = './various_rasters/africa.tif'#Path to reference file'
rst = rasterio.open(rst_fn)

mat = gdal_array.LoadFile(rst_fn)
sss=mat.shape


meta = rst.meta.copy()
meta.update(compress='lzw')


openShape = ogr.Open("./continents/continent.shp")
layers = openShape.GetLayer()
continents=[]
for element in layers:
	sh=loads(element.GetGeometryRef().ExportToWkb())
	continents.append(sh)




africa=continents[3].buffer(0)







sf = shapefile.Reader("./PA/WDPA_Jan2017-shapefile-polygons.dbf")
fields=sf.fields
records = sf.records()
ccc=[]
for i in records:
	ccc.append(i)




PA_dict=dict([['Not Reported',8],['Not Applicable',9],['Not Assigned',10],['Ia',1],['Ib',2],['II',3],['III',4],['IV',5],['V',6],['VI',7]])



openShape = ogr.Open("./PA/WDPA_Jan2017-shapefile-polygons.shp")
layers = openShape.GetLayer()
GEOM_parks=[]
sc=0
for element in layers:
	try:
		sh=loads(element.GetGeometryRef().ExportToWkb())
		pa_cat=PA_dict[ccc[sc][8]]
		try:	
			for pol in sh:
				if pol.intersects(africa):			
					GEOM_parks.append([convert_km(pol),pa_cat])
		except:
			if sh.intersects(africa):
				GEOM_parks.append([convert_km(sh),pa_cat])
	except:
		pass
	sc+=1
	if sc%1000==0:
		print sc,len(GEOM_parks)



out_fn='./various_rasters/protected_areas.tif'

out=rasterio.open(out_fn, 'w', **meta)
out_arr = out.read(1)
burned = features.rasterize(shapes=GEOM_parks, fill=0, out=out_arr, transform=out.transform)
out.write_band(1, burned)
out.close()


###OP CONCESSIONS
sc=0
openShape = ogr.Open("./gfw_oil_palm/gfw_oil_palm.shp")
layers = openShape.GetLayer()
GEOM_conc=[]
for element in layers:
	sh=loads(element.GetGeometryRef().ExportToWkb())
	try:	
		for pol in sh:
			if pol.intersects(africa):			
				GEOM_conc.append([convert_km(pol),1])
	except:
		if sh.intersects(africa):
			GEOM_conc.append([convert_km(sh),1])
	sc+=1
	if sc%1000==0:
		print sc



out_fn='./various_rasters/po_conc.tif'

out=rasterio.open(out_fn, 'w', **meta)
out_arr = out.read(1)
burned = features.rasterize(shapes=GEOM_conc, fill=0, out=out_arr, transform=out.transform)
out.write_band(1, burned)
out.close()





#MAKE PRIMATE RASTERS




#open IUCN mammal range dbf file; note that primates are the entries with 'PRIMATES' in field number 18	
sf = shapefile.Reader("./IUCN_terrestrial_mammals_ranges/TERRESTRIAL_MAMMALS.dbf")
fields=sf.fields
records = sf.records()
ccc=[]
for i in records:
	ccc.append(i)



#make dict spp, iucn status
prim_iucn_status_dict = []
for i in ccc:
	if i[18] == 'PRIMATES':
		prim_iucn_status_dict.append([i[1],i[22]])



prim_iucn_status_dict = dict(prim_iucn_status_dict)


#create a dictionary to convert IUCN threat values into numerical values, according to a geometric progression; data deficient (DD) records are given conservatively the same value as least concern (LC) ones

iucn=dict([['DD',2],['LC',2],['NT',4],['VU',8],['EN',16],['CR',32]])


#primate shapes are stored in a list called GEOM_PRI, together with primate species names and converted iucn status
openShape = ogr.Open("./IUCN_terrestrial_mammals_ranges/TERRESTRIAL_MAMMALS.shp")
layers = openShape.GetLayer()
GEOM_PRI=[]
sc=0
for element in layers:
	if ccc[sc][18]=='PRIMATES':
		sh=loads(element.GetGeometryRef().ExportToWkb())
		iucn_sc=iucn[ccc[sc][22]]
		try:
			for pol in sh:
				if pol.intersects(africa):
					GEOM_PRI.append([convert_km(pol),ccc[sc][1],iucn_sc])
		except:
			if sh.intersects(africa):
				GEOM_PRI.append([convert_km(sh),ccc[sc][1],iucn_sc])
	sc+=1
	print sc,len(GEOM_PRI)








###A few small ranged species resulted in empty tifs...doublecheck; perhaps add all_touched = True to features.rasterize; perhaps are coastal species? 

spp=sorted(list(set([i[1] for i in GEOM_PRI])))

m0=zeros(sss)
for i in spp:
	out_fn='./primate_rasters/'+i+'.tif'
	geom=[[j[0],j[2]] for j in GEOM_PRI if j[1]==i]
	out=rasterio.open(out_fn, 'w', **meta)
	out_arr = out.read(1)
	burned = features.rasterize(shapes=geom, fill=0, out=out_arr, transform=out.transform)
	out.write_band(1, burned)
	out.close()
	mat = gdal_array.LoadFile('./primate_rasters/'+i+'.tif')
	m0+=mat
	print i





out=rasterio.open('./various_rasters/primate_vulnerability.tif', 'w', **meta)
out.write(m0.astype(rasterio.uint8),1)
out.close()







#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
###COMPUTE AREAS OF COMPROMISE
#####make combined mask


mat_land_use = gdal_array.LoadFile('./various_rasters/land_use.tif')
mat_land_use[where(mat_land_use==40)]=0
mat_land_use[where(mat_land_use==50)]=0
mat_land_use[where(mat_land_use==60)]=0
mat_land_use[where(mat_land_use==70)]=0
mat_land_use[where(mat_land_use==80)]=0
mat_land_use[where(mat_land_use==81)]=0
mat_land_use[where(mat_land_use==200)]=0
mat_land_use[where(mat_land_use==255)]=0
mat_land_use[where(mat_land_use==90)]=1
mat_land_use[where(mat_land_use!=0)]=1

'''
#forest_mask
fmask = gdal_array.LoadFile('./various_rasters/land_use.tif')
fmask[where(fmask==40)]=0
fmask[where(fmask==50)]=0
fmask[where(fmask==60)]=0
fmask[where(fmask==70)]=0
fmask[where(fmask==80)]=0
fmask[where(fmask==81)]=0
fmask[where(fmask==200)]=0
fmask[where(fmask==255)]=0
fmask[where(fmask==90)]=0
fmask[where(fmask==20)]=0
fmask[where(fmask==30)]=0
fmask[where(fmask!=0)]=1
'''






mat_pa = gdal_array.LoadFile('./various_rasters/protected_areas.tif')
mat_conc = gdal_array.LoadFile('./various_rasters/po_conc.tif')




mat_suit = gdal_array.LoadFile('./various_rasters/suitability.tif')
mat_suit*=(mat_suit<8)	#9 is water, 8 is not suitable
mat_suit*=mat_land_use
mat_suit[where(mat_conc>0)]=0
mat_suit[where(mat_suit>0)]=8-mat_suit[where(mat_suit>0)]




mat_suit_low = gdal_array.LoadFile('./various_rasters/suitability_low.tif')
mat_suit_low*=(mat_suit_low<8)	#9 is water, 8 is not suitable
mat_suit_low*=mat_land_use
mat_suit_low[where(mat_conc>0)]=0
mat_suit_low[where(mat_suit_low>0)]=8-mat_suit_low[where(mat_suit_low>0)]



mat_suit_high = gdal_array.LoadFile('./various_rasters/suitability_high.tif')
mat_suit_high*=(mat_suit_high<8)	#9 is water, 8 is not suitable
mat_suit_high*=mat_land_use
mat_suit_high[where(mat_conc>0)]=0
mat_suit_high[where(mat_suit_high>0)]=8-mat_suit_high[where(mat_suit_high>0)]





mat_vuln = gdal_array.LoadFile('./various_rasters/primate_vulnerability.tif')
#mat_vuln*=fmask #filter by forest
mat_vuln_low=mat_vuln*(mat_suit_low>0)
mat_vuln_high=mat_vuln*(mat_suit_high>0)

mat_vuln=mat_vuln*(mat_suit>0)

def log_plus(x):
	return log(x+1)



log_plus_mat=vectorize(log_plus)

mat_vuln=log_plus_mat(mat_vuln)
mat_vuln_low=log_plus_mat(mat_vuln_low)
mat_vuln_high=log_plus_mat(mat_vuln_high)



#GAEZ suitability categories
def get_col(suit,vuln):
	if suit==0:
		return 0		
	elif suit>=5 and vuln<2:
		return 1
	elif 3<=suit<5 and vuln<2:
		return 2
	elif suit<3 and vuln<2:
		return 3
	elif suit>=5 and 2<=vuln<4:
		return 4
	elif 3<=suit<5 and 2<=vuln<4:
		return 5
	elif suit<=3 and 2<=vuln<4:
		return 6
	elif suit>=5 and vuln>=4:
		return 7	
	elif 3<=suit<5  and vuln>=4:
		return 8	
	elif suit<3 and vuln>=4:
		return 9
	else:
		return 0


get_col_mat=vectorize(get_col)




comp_dict=dict([[0,['#ffffff','not_suitable']],[1,['#006d2c','HL']],[2,['#31a354','ML']],[3,['#bae4b3','LL']],[4,['#2b8cbe','HM']],[5,['#a6bddb','MM']],[6,['#ece7f2','LM']],[7,['#a50f15','HH']],[8,['#de2d26','MH']],[9,['#fcae91','LH']],[10,['#ff00ff','PA']]])



res_mat=get_col_mat(mat_suit,mat_vuln)
res_mat[where(mat_pa!=0)]=10
res_mat*=(mat_suit>0)


out=rasterio.open('./various_rasters/areas_of_compromise.tif', 'w', **meta)
out.write(res_mat.astype(rasterio.uint8),1)
out.close()



res_mat_low=get_col_mat(mat_suit_low,mat_vuln_low)
res_mat_low[where(mat_pa!=0)]=10
res_mat_low*=(mat_suit_low>0)

out=rasterio.open('./various_rasters/areas_of_compromise_low.tif', 'w', **meta)
out.write(res_mat_low.astype(rasterio.uint8),1)
out.close()


res_mat_high=get_col_mat(mat_suit_high,mat_vuln_high)
res_mat_high[where(mat_pa!=0)]=10
res_mat_high*=(mat_suit_high>0)

out=rasterio.open('./various_rasters/areas_of_compromise_high.tif', 'w', **meta)
out.write(res_mat_high.astype(rasterio.uint8),1)
out.close()





#####COMPUTE AREAS KM2
comp_dict=dict([[0,['#ffffff','not_suitable']],[1,['#006d2c','HL']],[2,['#31a354','ML']],[3,['#bae4b3','LL']],[4,['#2b8cbe','HM']],[5,['#a6bddb','MM']],[6,['#ece7f2','LM']],[7,['#a50f15','HH']],[8,['#de2d26','MH']],[9,['#fcae91','LH']],[10,['#ff00ff','PA']]])


out=open('./results/comparison_aoc_low_int_high.csv','w')
out.write('category,color,km2_low,km2_intermediate,km2_high\n')

for i in range(11):
	km2_int=100*(res_mat==i).sum()
	km2_low=100*(res_mat_low==i).sum()
	km2_high=100*(res_mat_high==i).sum()
	col,name=comp_dict[i]
	out.write(','.join(map(str,[name,col,km2_low,km2_int,km2_high]))+'\n')



out.close()




#####
#Land Conversion Scenarios
######


mat_carbon = gdal_array.LoadFile('./various_rasters/carbon.tif')
mat_acc = gdal_array.LoadFile('./various_rasters/accessibility.tif')

mat_land_use = gdal_array.LoadFile('./various_rasters/land_use.tif')
mat_land_use[where(mat_land_use==40)]=0
mat_land_use[where(mat_land_use==50)]=0
mat_land_use[where(mat_land_use==60)]=0
mat_land_use[where(mat_land_use==70)]=0
mat_land_use[where(mat_land_use==80)]=0
mat_land_use[where(mat_land_use==81)]=0
mat_land_use[where(mat_land_use==200)]=0
mat_land_use[where(mat_land_use==255)]=0
mat_land_use[where(mat_land_use==90)]=1
mat_land_use[where(mat_land_use!=0)]=1


mat_pa = gdal_array.LoadFile('./various_rasters/protected_areas.tif')
mat_conc = gdal_array.LoadFile('./various_rasters/po_conc.tif')
mat_suit = gdal_array.LoadFile('./various_rasters/suitability.tif')

mat_suit*=(mat_suit<8)	#9 is water, 8 is not suitable; this time, leave 1 as most suitable
mat_suit*=mat_land_use
mat_suit[where(mat_conc>0)]=0
mat_suit[where(mat_pa>0)]=0		


mat_vuln = gdal_array.LoadFile('./various_rasters/primate_vulnerability.tif')
mat_vuln=mat_vuln*(mat_suit>0)




R,C=mat_suit.shape


vals=[]
for i in range(R):
	for j in range(C):
		if mat_suit[i][j]>0:
			vals.append([mat_suit[i][j],mat_acc[i][j],mat_carbon[i][j],mat_vuln[i][j],i*C+j])
	print R-i

	





vals_ok=[i[:4] for i in vals]	#suit,acc,carbon
indexes=dict([[vals[i][-1],i] for i in range(len(vals))])

spp = sorted(list(set([i[:-4].split('_')[0] for i in listdir('./primate_rasters/')])))

vals_spp=[[] for i in range(len(vals))]
spp_ranges = []
out = open('./results/range_adjustment.csv','w')
out.write('species,iucn_range,range_adjusted\n')
for sp in range(len(spp)):
	row=[]
	mat = gdal_array.LoadFile('./primate_rasters/'+spp[sp]+'.tif')
	src=rasterio.open('./primate_rasters/'+spp[sp]+'.tif')
	mat=(mat>0)*1
	range_iucn = mat.sum()*100
	mat*=mat_land_use
	mat*=(mat_conc==0)
	range_adjusted = mat.sum()*100.0
	spp_ranges.append(mat.sum()*100.0)
	mat*=(mat_pa==0)
	#range_protected = range_adjusted - mat.sum()*100
	out.write(','.join(map(str,[spp[sp],range_iucn,range_adjusted]))+'\n')
	mat=mat.astype(numpy.int32,copy=False)
	mat*=(mat_suit>0)
	occ=where(mat>0)
	occ_=occ[0]*C+occ[1]
	for j in occ_:
		vals_spp[indexes[j]].append(sp)
	print spp[sp]
	src.close()


out.close()



spp_ranges=array(spp_ranges)
ranges_0=mean(spp_ranges)

spp_iucn_status = [prim_iucn_status_dict[i] for i in spp]

####EXPLORE ALL SCENARIOS

var_names=['suitability','accessibility','carbon','vulnerability']

all_mods=[list(i) for i in list(itertools.permutations([0,1,2],1))+list(itertools.permutations([0,1,2],2))+list(itertools.permutations([0,1,2],3))]
all_mods+=[[3],[]]


for model in all_mods:
	if model!=[]:
		mod_name='_'.join([var_names[j] for j in model])
	else:
		mod_name='random'
	vals=[[vals_ok[k][j] for j in model]+[random()]+[vals_spp[k]] for k in range(len(vals_ok))]
	out=open('./results/res_'+mod_name+'.csv','w')
	for rep in range(1000):	#we performed 1000 replicates in the actual analyses
		if rep>0:
			for i in range(len(vals)):
				vals[i][len(model)]=random()
		vals.sort()
		ranges_=array(deepcopy(spp_ranges))
		lost_area=0.0
		out.write(','.join(map(str,[0.0,0.0,0.0,0.0]))+'\n')
		sc=0
		for i in vals[:5501]:
			ranges_[i[-1]]-=100.0
			lost_area+=100.0
			if sc%10 == 0:
				res=[lost_area,(spp_ranges-ranges_).sum(),(spp_ranges-ranges_).mean(),len([j for j in ranges_/spp_ranges if j<=0.9]),len([j for j in ranges_/spp_ranges if j<=0.95]),len([j for j in ranges_/spp_ranges if j<1])]
				out.write(','.join(map(str,res))+'\n')
			if sc == 2200:
				out_prim_ranges=open('./results/'+mod_name+'_primate_range_loss_food_half_africa.csv','a')
				for prim_sp in range(len(spp)):
					out_prim_ranges.write(','.join(map(str,[spp[prim_sp],spp_iucn_status[prim_sp],spp_ranges[prim_sp],ranges_[prim_sp]]))+'\n')
				out_prim_ranges.close()
			if sc == 2650:
				out_prim_ranges=open('./results/'+mod_name+'_primate_range_loss_food_biofuel_half_africa.csv','a')
				for prim_sp in range(len(spp)):
					out_prim_ranges.write(','.join(map(str,[spp[prim_sp],spp_iucn_status[prim_sp],spp_ranges[prim_sp],ranges_[prim_sp]]))+'\n')
				out_prim_ranges.close()
			if sc == 4400:
				out_prim_ranges=open('./results/'+mod_name+'_primate_range_loss_food_all_africa.csv','a')
				for prim_sp in range(len(spp)):
					out_prim_ranges.write(','.join(map(str,[spp[prim_sp],spp_iucn_status[prim_sp],spp_ranges[prim_sp],ranges_[prim_sp]]))+'\n')
				out_prim_ranges.close()
			if sc == 5300:	
				out_prim_ranges=open('./results/'+mod_name+'_primate_range_loss_food_biofuel_all_africa.csv','a')
				for prim_sp in range(len(spp)):
					out_prim_ranges.write(','.join(map(str,[spp[prim_sp],spp_iucn_status[prim_sp],spp_ranges[prim_sp],ranges_[prim_sp]]))+'\n')
				out_prim_ranges.close()
			sc+=1
		res=[lost_area,(spp_ranges-ranges_).sum(),(spp_ranges-ranges_).mean(),len([j for j in ranges_/spp_ranges if j<=0.9]),len([j for j in ranges_/spp_ranges if j<=0.95]),len([j for j in ranges_/spp_ranges if j<1])]
		out.write(','.join(map(str,res))+'\n')
		print rep
	out.close()






####OPTIMIZATION SCENARIO for FIG S5

vals_a=array(vals_ok)

mmm=vals_a.min(0)
MMM=vals_a.max(0)
ddd=MMM-mmm

vals=[]
sc=0
for i in vals_a:
	vals.append([((MMM-i)/ddd).sum(),random()]+[vals_spp[sc]])		#1 = most suitable, most accessible, smallest carbon, smallest vulnerability
	sc+=1






out=open('./results/res_optimization.csv','w')
for rep in range(1000):
	if rep>0:
		for i in range(len(vals)):
			vals[i][1]=random()
	vals.sort(reverse=True)
	ranges_=array(deepcopy(spp_ranges))
	lost_area=0.0
	sc=0
	for i in vals[:5501]:
		ranges_[i[-1]]-=100.0
		lost_area+=100.0
		if sc%10 == 0:
			res=[lost_area,(spp_ranges-ranges_).sum(),(spp_ranges-ranges_).mean(),len([j for j in ranges_/spp_ranges if j<=0.9]),len([j for j in ranges_/spp_ranges if j<=0.95]),len([j for j in ranges_/spp_ranges if j<1])]
			out.write(','.join(map(str,res))+'\n')
		if sc == 2200:
			out_prim_ranges=open('./results/optimization_primate_range_loss_food_half_africa.csv','a')
			for prim_sp in range(len(spp)):
				out_prim_ranges.write(','.join(map(str,[spp[prim_sp],spp_iucn_status[prim_sp],spp_ranges[prim_sp],ranges_[prim_sp]]))+'\n')
			out_prim_ranges.close()
		if sc == 2650:
			out_prim_ranges=open('./results/optimization_primate_range_loss_food_biofuel_half_africa.csv','a')
			for prim_sp in range(len(spp)):
				out_prim_ranges.write(','.join(map(str,[spp[prim_sp],spp_iucn_status[prim_sp],spp_ranges[prim_sp],ranges_[prim_sp]]))+'\n')
			out_prim_ranges.close()
		if sc == 4400:
			out_prim_ranges=open('./results/optimization_primate_range_loss_food_all_africa.csv','a')
			for prim_sp in range(len(spp)):
				out_prim_ranges.write(','.join(map(str,[spp[prim_sp],spp_iucn_status[prim_sp],spp_ranges[prim_sp],ranges_[prim_sp]]))+'\n')
			out_prim_ranges.close()
		if sc == 5300:
			out_prim_ranges=open('./results/optimization_primate_range_loss_food_biofuel_all_africa.csv','a')
			for prim_sp in range(len(spp)):
				out_prim_ranges.write(','.join(map(str,[spp[prim_sp],spp_iucn_status[prim_sp],spp_ranges[prim_sp],ranges_[prim_sp]]))+'\n')
			out_prim_ranges.close()	
		sc+=1
	res=[lost_area,(spp_ranges-ranges_).sum(),(spp_ranges-ranges_).mean(),len([j for j in ranges_/spp_ranges if j<=0.9]),len([j for j in ranges_/spp_ranges if j<=0.95]),len([j for j in ranges_/spp_ranges if j<1])]
	out.write(','.join(map(str,res))+'\n')
	print rep



out.close()





###MAKE FIGS 1-2


###vectorize areas_of_compromise raster for mapping purposes
mask = None
with rasterio.drivers():
	with rasterio.open('./various_rasters/areas_of_compromise.tif') as src:
		image = src.read(1) # first band
		results = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.affine)))




geoms = list(results)


GEOMS=[]
for i in geoms:
	GEOMS.append([shape(i['geometry']),comp_dict[i['properties']['raster_val']]])





#save the map to shapefile, with colors to be attributed to each category 
schema = {'geometry': 'Polygon','properties': {'color': 'str','sv': 'str'},}
with fiona.open('./results/areas_of_compromise.shp', 'w', 'ESRI Shapefile', schema) as c:
	for square in GEOMS:
		try:
			for pol in square[0]:
				c.write({'geometry': mapping(convert_wgs84(pol)),'properties': {'color': square[1][0],'sv': square[1][1]},})
		except:
			c.write({'geometry': mapping(convert_wgs84(square[0])),'properties': {'color': square[1][0],'sv': square[1][1]},})




####Do the same for the primate vulnerability and the suitability map
###primate cumulative vulnerability


mask = None
with rasterio.drivers():
	with rasterio.open('./various_rasters/primate_vulnerability.tif') as src:
		image = src.read(1) # first band
		#image.dtype='int32'
		results = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.affine)))




geoms = list(results)


GEOMS=[]
for i in geoms:
	GEOMS.append([shape(i['geometry']),i['properties']['raster_val']])





#save the map to shapefile, with colors to be attributed to each category 
schema = {'geometry': 'Polygon','properties': {'sv': 'float'},}
with fiona.open('./results/primate_vulnerability.shp', 'w', 'ESRI Shapefile', schema) as c:
	for square in GEOMS:
		try:
			for pol in square[0]:
				c.write({'geometry': mapping(convert_wgs84(pol)),'properties': {'sv': square[1]},})
		except:
			c.write({'geometry': mapping(convert_wgs84(square[0])),'properties': {'sv': square[1]},})



####oil palm suitability


mask = None
with rasterio.drivers():
	with rasterio.open('./various_rasters/suitability.tif') as src:
		image = src.read(1) # first band
		#image.dtype='int32'
		results = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.affine)))




geoms = list(results)


GEOMS=[]
for i in geoms:
	GEOMS.append([shape(i['geometry']),i['properties']['raster_val']])


GEOMS=[i for i in GEOMS if 0<i[1]<8]


#save the map to shapefile, with colors to be attributed to each category 
schema = {'geometry': 'Polygon','properties': {'sv': 'float'},}
with fiona.open('./results/oil_palm_suitability.shp', 'w', 'ESRI Shapefile', schema) as c:
	for square in GEOMS:
		try:
			for pol in square[0]:
				c.write({'geometry': mapping(convert_wgs84(pol)),'properties': {'sv': square[1]},})
		except:
			c.write({'geometry': mapping(convert_wgs84(square[0])),'properties': {'sv': square[1]},})







####MAKE PRIMATE CUMULATIVE VULNERABILITY MAP FOR FIG 1a


m = Basemap(resolution='l',area_thresh=100000.,projection='cyl',llcrnrlat=-35, llcrnrlon=-20,urcrnrlat=38, urcrnrlon=52)
m.fillcontinents(color='lightgrey',lake_color='lightgrey')
m.drawcoastlines(linewidth=0.1)



shpname='./results/primate_vulnerability'
m.readshapefile(shpname,shpname,drawbounds=False)
info=shpname+'_info'
patches=[]
vals=[]
m_log=6
for xy, info in zip(getattr(m,shpname), getattr(m,info)):
	if float(info['sv'])>0:
		col_val=int(round(255*(log(float(info['sv'])+1)/m_log)))
		#print col_val	
		col=plasma(col_val)
		poly = Polygon(xy, facecolor=col,ec=col,alpha=1,linewidth=0.0)#,fill=True,joinstyle='round',rasterized=True)
		plt.gca().add_patch(poly)







cmleg = zeros((1,7))
for i in range(7):
    cmleg[0,i] = i



cbar=plt.imshow(cmleg, cmap=plasma, aspect=1)
plt.colorbar()



plt.savefig('./results/primate_vulnerability.pdf',dpi=300)





####MAKE OIL PALM SUITABILITY MAP FOR FIG. 1b

clear=[plt.clf() for i in range(100000)]
m = Basemap(resolution='l',area_thresh=100000.,projection='cyl',llcrnrlat=-35, llcrnrlon=-20,urcrnrlat=38, urcrnrlon=52)
m.fillcontinents(color='lightgrey',lake_color='lightgrey')
m.drawcoastlines(linewidth=0.1)




shpname='./results/oil_palm_suitability'
m.readshapefile(shpname,shpname,drawbounds=False)
info=shpname+'_info'
patches=[]
for xy, info in zip(getattr(m,shpname), getattr(m,info)):
		col=plasma(int(round(255*((8-float(info['sv']))/7.0))))
		poly = Polygon(xy, facecolor=col,ec=col,alpha=1,linewidth=0.0)#,fill=True,joinstyle='round',rasterized=True)
		plt.gca().add_patch(poly)






cmleg = zeros((1,8))
for i in range(1,8):
    cmleg[0,i] = i



cbar=plt.imshow(cmleg, cmap=plasma, aspect=1)
plt.colorbar()



plt.savefig('./results/oil_palm_suitability.pdf',dpi=300)


###MAKE MAP OF AREAS OF COMPROMISE FOR FIG.2
clear=[plt.clf() for i in range(100000)]


matplotlib.rcParams['hatch.linewidth'] = 0.001

m = Basemap(resolution='l',area_thresh=100000.,projection='cyl',llcrnrlat=-28, llcrnrlon=-20,urcrnrlat=13, urcrnrlon=52)
m.fillcontinents(color='darkgrey',lake_color='darkgrey')
m.drawcoastlines(linewidth=0.1)
m.drawcountries(linewidth=0.3,color='white')



shpname='./results/areas_of_compromise'		#this file is generated by script.py
m.readshapefile(shpname,shpname,drawbounds=False)
info=shpname+'_info'
for xy, info in zip(getattr(m,shpname), getattr(m,info)):
	if info['color'] not in ['#ffffff','#ff00ff']:
		poly = Polygon(xy, facecolor=info['color'],ec='lightgrey',alpha=1,linewidth=0.0,fill=True)#hatch='//////')#,joinstyle='round',rasterized=False)
		plt.gca().add_patch(poly)


m.readshapefile(shpname,shpname,drawbounds=False)#this ensures protected areas are plot on top of the image.
info=shpname+'_info'
for xy, info in zip(getattr(m,shpname), getattr(m,info)):
	if info['color']=='#ff00ff':
		poly = Polygon(xy, facecolor=info['color'],ec='lightgrey',alpha=1,linewidth=0.0,fill=True)#hatch='//////')#,joinstyle='round',rasterized=False)
		plt.gca().add_patch(poly)




plt.savefig('./results/areas_of_compromise.pdf',dpi=300)

clear=[plt.clf() for i in range(100000)]



###for figure S1


mat_land_use = gdal_array.LoadFile('./various_rasters/land_use.tif')
mat_land_use[where(mat_land_use==40)]=0
mat_land_use[where(mat_land_use==50)]=0
mat_land_use[where(mat_land_use==60)]=0
mat_land_use[where(mat_land_use==70)]=0
mat_land_use[where(mat_land_use==80)]=0
mat_land_use[where(mat_land_use==81)]=0
mat_land_use[where(mat_land_use==200)]=0
mat_land_use[where(mat_land_use==255)]=0
mat_land_use[where(mat_land_use==90)]=1
mat_land_use[where(mat_land_use!=0)]=1




mat_pa = gdal_array.LoadFile('./various_rasters/protected_areas.tif')
mat_conc = gdal_array.LoadFile('./various_rasters/po_conc.tif')




mat_suit = gdal_array.LoadFile('./various_rasters/suitability.tif')
mat_suit*=(mat_suit<8)	#9 is water, 8 is not suitable
mat_suit*=mat_land_use
mat_suit[where(mat_conc>0)]=0
mat_suit[where(mat_suit>0)]=8-mat_suit[where(mat_suit>0)]




mat_vuln = gdal_array.LoadFile('./various_rasters/primate_vulnerability.tif')

mat_vuln=mat_vuln*(mat_suit>0)



list_vuln=mat_vuln.flatten()
list_suit=mat_suit.flatten()

out=open('./results/suitability_vs_vulnerability.csv','w')
for i in range(len(list_suit)):
	out.write(','.join(map(str,[list_suit[i],list_vuln[i]]))+'\n')


out.close()





####


#####SENSITIVITY ANALYSIS OF PRIMATE RANGES
rst_fn = './various_rasters/africa.tif'#Path to reference file'
rst = rasterio.open(rst_fn)

mat = gdal_array.LoadFile(rst_fn)
sss=mat.shape





mat_land_use = gdal_array.LoadFile('./various_rasters/land_use.tif')
mat_land_use[where(mat_land_use==40)]=0
mat_land_use[where(mat_land_use==50)]=0
mat_land_use[where(mat_land_use==60)]=0
mat_land_use[where(mat_land_use==70)]=0
mat_land_use[where(mat_land_use==80)]=0
mat_land_use[where(mat_land_use==81)]=0
mat_land_use[where(mat_land_use==200)]=0
mat_land_use[where(mat_land_use==255)]=0
mat_land_use[where(mat_land_use==90)]=1
mat_land_use[where(mat_land_use!=0)]=1



mat_pa = gdal_array.LoadFile('./various_rasters/protected_areas.tif')
mat_conc = gdal_array.LoadFile('./various_rasters/po_conc.tif')




mat_suit = gdal_array.LoadFile('./various_rasters/suitability.tif')
mat_suit*=(mat_suit<8)	#9 is water, 8 is not suitable
mat_suit*=mat_land_use
mat_suit[where(mat_conc>0)]=0
mat_suit[where(mat_pa!=0)]=0
mat_suit_mask = (mat_suit>0)*1


def log_plus(x):
	return log(x+1)


def get_vuln_class(vuln):
	if 0<vuln<2:
		return 1
	elif 2<=vuln<4:
		return 2
	elif vuln>=4:
		return 2
	else:
		return 0


get_vuln_class_mat=vectorize(get_vuln_class)



log_plus_mat=vectorize(log_plus)


spp = [i.split('.')[0] for i in listdir('./primate_rasters') if 'adjusted' not in i]

m_0=zeros(sss)
for i in spp:
	mat = gdal_array.LoadFile('./primate_rasters/'+i+'.tif')
	m_0+=mat


m_0*=mat_suit_mask
mat_vuln_0 = log_plus_mat(m_0)
mat_vuln_class_0 = get_vuln_class_mat(mat_vuln_0) 

suit_cells_n=float((mat_vuln_class_0>0).sum())

from numpy import arange
out = open('./results/sensitivity_vuln_index.csv','a')
for eee in arange(0.01,0.51,0.01):
	m0=zeros(sss)
	for i in spp:
		mat = gdal_array.LoadFile('./primate_rasters/'+i+'.tif')
		xxx = where(mat>0)
		l = len(xxx[0])
		to_del = sample(range(l),int(round(l*eee)))
		for x in to_del:
			mat[xxx[0][x],xxx[1][x]] = 0	
		m0+=mat
	m0*=mat_suit_mask
	mat_vuln=log_plus_mat(m0)
	mat_vuln_class = get_vuln_class_mat(mat_vuln)
	out.write(','.join(map(str,[eee,((mat_vuln_class_0!=mat_vuln_class)*(mat_vuln_class_0>0)).sum()/suit_cells_n
]))+'\n')



out.close()




###reduction of ranges through land use map


mat_land_use = gdal_array.LoadFile('./various_rasters/land_use.tif')
mat_land_use[where(mat_land_use==40)]=0
mat_land_use[where(mat_land_use==50)]=0
mat_land_use[where(mat_land_use==60)]=0
mat_land_use[where(mat_land_use==70)]=0
mat_land_use[where(mat_land_use==80)]=0
mat_land_use[where(mat_land_use==81)]=0
mat_land_use[where(mat_land_use==200)]=0
mat_land_use[where(mat_land_use==255)]=0
mat_land_use[where(mat_land_use==90)]=1
mat_land_use[where(mat_land_use!=0)]=1



rst_fn = './various_rasters/africa.tif'#Path to reference file'
rst = rasterio.open(rst_fn)

mat = gdal_array.LoadFile(rst_fn)
sss=mat.shape


meta = rst.meta.copy()
meta.update(compress='lzw')
#meta.update(dtype='uint32')



spp = [i.split('.')[0] for i in listdir('./primate_rasters')]
res = []
for i in spp:
	mat = gdal_array.LoadFile('./primate_rasters/'+i+'.tif')
	#mat *= (mat_suit>0)
	iucn_range = (mat>0).sum()
	mat *= mat_land_use
	filt_range = (mat>0).sum()
	res.append([i, iucn_range, filt_range])
	print i,filt_range/float(iucn_range)
	out=rasterio.open('./primate_rasters/'+i+'_adjusted.tif', 'w', **meta)
	out.write(mat.astype(rasterio.uint8),1)
	out.close()




out=open('./results/iucn_range_reduction_land_use.csv','a')
for i in res:
	out.write(','.join(map(str,i))+'\n')


out.close()




#####MAKE MAPS OF PRIMATES' RANGE REDUCTION BY LAND USE FILTERING



colors = ['#ffffff','#000000']

color_map = ListedColormap(colors)


def convertXY(xy_source, inproj, outproj):
    # function to convert coordinates
    shape = xy_source[0, :, :].shape
    size = xy_source[0, :, :].size
    # the ct object takes and returns pairs of x,y, not 2d grids
    # so the the grid needs to be reshaped (flattened) and back.
    ct = osr.CoordinateTransformation(inproj, outproj)
    xy_target = np.array(ct.TransformPoints(xy_source.reshape(2, size).T))
    xx = xy_target[:, 0].reshape(shape)
    yy = xy_target[:, 1].reshape(shape)
    return xx, yy

# Read the data and metadata


fff = [i for i in listdir('./primate_rasters') if i[-3:] == 'tif']
for sp in fff:
	ds = gdal.Open("./primate_rasters/"+sp)
	gt = ds.GetGeoTransform()
	proj = ds.GetProjection()
	data = ds.ReadAsArray()
	#data = np.flipud(data)
	xres = gt[1]
	yres = gt[5]
	xmin = gt[0] + xres * 0.5
	xmax = gt[0] + (xres * data.shape[1]) - xres * 0.5
	ymin = gt[3] + (yres * data.shape[0]) - yres * 0.5
	ymax = gt[3] + yres * 0.5
	ds = None
	xy_source = np.mgrid[xmin:xmax + xres:xres, ymax:ymin + yres:yres]
	in_proj = osr.SpatialReference()
	in_proj.ImportFromWkt(proj)
	out_proj = osr.SpatialReference()
	out_proj.ImportFromEPSG(4326)
	xx, yy = convertXY(xy_source, in_proj, out_proj)
	mlon,mlat,Mlon,Mlat = xx.min(),yy.min(),xx.max(),yy.max()
	plt.rcParams['hatch.linewidth'] = 0.1
	m = Basemap(resolution='l', area_thresh=100000., projection='cyl',llcrnrlat=-35, llcrnrlon=-20,urcrnrlat=35, urcrnrlon=52)
	m.drawcoastlines(linewidth=0.1)
	xx_m, yy_m = m(xx, yy)  # Convert latlong coordinates to basemap proj
	im = m.pcolormesh(xx_m, yy_m, data.T, cmap=color_map)
	plt.title(sp[:-4].split('_')[0])
	plt.savefig("./primate_range_reduction_figures/"+sp[:-3]+"png", dpi=300)
	clear=[plt.clf() for i in range(100000)]







###########
from load import csv_
from numpy import mean
fff = []
for scen in ['vulnerability','suitability','accessibility','carbon','random']:
	fff.append(csv_('./results/'+scen+'_primate_range_loss_food_biofuel_half_africa.csv'))




spp=sorted(list(set([i[0] for i in fff[0] if float(i[2])>0])))
res=[]
for sp in spp:
	ROW=[]
	for ff in range(5):
		row=[]
		for j in fff[ff]:
			if sp==j[0]:
				status = j[1]
				row.append(1-float(j[3])/float(j[2]))
		ROW.append(mean(row))	
	res.append([sp,status]+ROW)
	print res[-1]




out=open('table_S1.csv','w')
out.write('species,IUCN_status,vulnerability,suitability,accessibility,carbon,random\n')
for i in res:
	out.write(','.join(map(str,[i[0],i[1],round(i[2]*100,1),round(i[3]*100,1),round(i[4]*100,1),round(i[5]*100,1),round(i[6]*100,1)]))+'\n')


out.close()




###MAKE FIGS 3-4 and supplementary figures

os.system("Rscript ./make_plots.R")

