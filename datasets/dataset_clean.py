import pickle
import gzip
import cv2


class Targets:
    def __init__(self, datasetPath):
        self.pickleFile = None

        if datasetPath != None:
            self.pickleFile = gzip.open(datasetPath, mode='ab')

    def parse(self, dct):
        if self.pickleFile != None:
            pickle.dump(dct, self.pickleFile)


def clean(num):
    file = open("dataset_clean_file/dataset"+str(num)+".pickle", "rb") # 设置数据集清理文件的位置
    data_list = pickle.load(file)
    raw_dataset = gzip.open("d:/no_traffic/dataset"+str(num)+".pz", "rb")  # 设置需要清理的数据集存储位置以及存储名称，默认按数字增序向后命名
    new_data_path = "dataset/dataset_example.pz"   # 设置清理后数据集要存储的位置
    i = 1
    d = 0
    nd = 0
    left = 0
    right = 0
    while True:
        try:
            data_dict = pickle.load(raw_dataset)
            if i in data_list:
                print("=========delete "+str(i)+" successfully========")
                d = d+1
                i = i+1
                continue
            if float(data_dict["steering"])>0.10:
                right = right+1
            elif float(data_dict["steering"])<-0.10:
                left = left+1
            print("run on "+str(num)+" dataset, work on "+str(i))
            target = Targets(datasetPath=new_data_path)
            target.parse(data_dict)
            i = i+1
            nd = nd+1

        except (EOFError,pickle.UnpicklingError):
            print("error/end")
            print("total: " + str(i))
            break


def clean_throw_after(num):
    file = open("cleaned_data/dataset" + str(num) + ".pickle", "rb")
    data_list = pickle.load(file)
    raw_dataset = gzip.open("dataset/dataset" + str(num) + ".pz", "rb")
    new_data_path = "D:/gta_datasets/dataset" + str(num) + ".pz"
    del_info = "Q:/code/ai_car/console/new_dataset/dataset" + str(num) + ".pickle"
    i = 0
    d = 0
    nd = 0
    while True:
        try:
            if i == 17000:
                print("error/end")
                print("total: " + str(i))
                ls = [num, i, d, nd]
                pickle.dump(ls, open(del_info, "wb"))
                break
            data_dict = pickle.load(raw_dataset)
            if i in data_list:
                print("=========delete " + str(i) + " successfully========")
                d = d + 1
                i = i + 1
                continue
            print("run on " + str(num) + " dataset, work on " + str(i))
            target = Targets(datasetPath=new_data_path)
            target.parse(data_dict)
            i = i + 1
            nd = nd + 1

        except EOFError:
            print("error/end")
            print("total: " + str(i))
            ls = [num, i, d, nd]
            pickle.dump(ls, open(del_info, "wb"))
            break


if __name__ == "__main__":
    list_i = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 需要清理的数据集序号，可一次清理多个数据集
    for i in list_i:
        clean(i)
        print("all end")
