import os
import sys
sys.path.insert(0,os.getcwd())
from utils.train_utils import get_info

def main():
    classes_path    = 'datas/annotations.txt'   # 标签文件位置
    datasets_path   = 'datasets'
    datasets        = ["train", "test"]
    classes, indexs = get_info(classes_path)    # 获取类名和索引，生成两个列表
    
    for dataset in datasets:
        txt_file = open('datas/' + dataset + '.txt', 'w')
        datasets_path_ = os.path.join(datasets_path, dataset)
        classes_name      = os.listdir(datasets_path_)
        
        for name in classes_name:
            if name not in classes:
                continue
            cls_id = indexs[classes.index(name)]    # list.index()用于找出列表中某一个值第一个匹配项的索引位置
            images_path = os.path.join(datasets_path_, name)
            images_name = os.listdir(images_path)
            for photo_name in images_name:
                _, postfix = os.path.splitext(photo_name)   # os.path.splitext(“文件路径”) 分离文件名与扩展名
                if postfix not in ['.jpg', '.png', '.jpeg','.JPG', '.PNG', '.JPEG']:
                    continue
                txt_file.write('%s'%(os.path.join(images_path, photo_name)) + ' ' + str(cls_id))
                txt_file.write('\n')
        txt_file.close()
if __name__ == "__main__":
    main()
