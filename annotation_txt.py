import os
import pyzjr as pz
# pip install pyzjr==1.1.1 --user -i https://pypi.tuna.tsinghua.edu.cn/simple
classes = ['cat', 'dog']
path = 'train'

if __name__ == '__main__':
    with open('class_data.txt', 'w') as txt_file:  # 打开文件，注意使用 'w' 模式
        file_list = [os.path.join(path, i) for i in os.listdir(path)]
        for data_path in file_list:
            types_name, _ = pz.getPhotopath(data_path, True)
            cls_id = classes.index(os.path.basename(data_path))
            for type_name in types_name:
                line = f"{str(cls_id)};{str(type_name)}"
                txt_file.write(line + '\n')  # 追加写入数据
