from data_processor.Proj5DatasetProcessor import Proj5DatasetProcessor

if __name__=='__main__':
    proj5_processor = Proj5DatasetProcessor()
    proj5_processor.dataset_generator_proj5_image('2020', file_name='proj5_allfire_img.npy')