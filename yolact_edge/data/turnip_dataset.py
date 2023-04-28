from config import dataset_base

label_map = {1:1}
CLASSES={'turnip'}

turnip_dataset = dataset_base.copy({
    'name': 'COCO_TURNIP',
   
    'train_images': 'path_to_training_images',
    'train_info':   'path_to_training_annotation',

    'valid_images': 'path_to_validation_images',
    'valid_info':   'path_to_validation_annotation',
    'class_names': CLASSES,
})