import glob
import os

if __name__=='__main__':
    file_list = glob.glob('data/infer/2023*.tif')
    for file in file_list:
        date = file.split('/')[-1].split('.')[0]
        os.system('gsutil cp '+file+' gs://ai4wildfire/VNPAFDL/CANADA/'+date+'.tif')
        os.system('earthengine upload image --force --time_start ' + date+'T00:00:00'+ ' --asset_id=projects/ee-eo4wildfire/assets/VNPAFDL_CANADA/' + date + ' --pyramiding_policy=sample gs://ai4wildfire/VNPAFDL/CANADA/'+date+'*.tif')
