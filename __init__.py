def classFactory(iface):
    from .geomorphons_plugin import GeomorphonsPlugin
    return GeomorphonsPlugin(iface)



from osgeo import gdal

def load_dem(file_path):
    dataset = gdal.Open(file_path)
    band = dataset.GetRasterBand(1)
    return band.ReadAsArray()