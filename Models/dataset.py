class Custom_Dataset(data.Dataset):
    def __init__(self, path, phase, mode):
        self.path = path
        self.phase = phase
        self.mode = mode

    def __getitem__(self, index):
        with h5py.File(self.path, 'r') as path:
            data = path["x" + str(self.phase)][index]
            label = path["y" + str(self.phase)][index]
            data = data / 255
            return data, label
    
    def __len__(self):
        with h5py.File(self.path, 'r') as path: 
            data = path["x" + str(self.phase)]
            return len(data)
        
test_tran_dataset = Custom_Dataset(path= r"/media/ram/338f6363-03b7-4ad7-a2be-40c31f59dee4/20230418_backup/ram/Students/B.Tech_2020/Madhav/mtnn/Datasets/NitrUAVCorridorV1/Translation(Angle))/TestTranslation(Angle).h5", phase= 'test', mode= "tran")
test_rot_dataset = Custom_Dataset(path= r"/media/ram/338f6363-03b7-4ad7-a2be-40c31f59dee4/20230418_backup/ram/Students/B.Tech_2020/Madhav/mtnn/Datasets/NitrUAVCorridorV1/Rotation(Distance)/TestRotation(Distance).h5", phase= 'test', mode="rot")

test_tran_loader = data.DataLoader(test_tran_dataset, batch_size=1)
test_rot_loader = data.DataLoader(test_rot_dataset, batch_size=1)

train_tran_dataset = Custom_Dataset(path= r"/media/ram/338f6363-03b7-4ad7-a2be-40c31f59dee4/20230418_backup/ram/Students/B.Tech_2020/Madhav/mtnn/Datasets/NitrUAVCorridorV1/Translation(Angle))/TrainTranslation(Angle).h5", phase= 'train', mode="tran")
train_rot_dataset = Custom_Dataset(path= r"/media/ram/338f6363-03b7-4ad7-a2be-40c31f59dee4/20230418_backup/ram/Students/B.Tech_2020/Madhav/mtnn/Datasets/NitrUAVCorridorV1/Rotation(Distance)/TrainRotation(Distance).h5", phase= 'train', mode="rot")

train_tran_loader = data.DataLoader(train_tran_dataset, batch_size=20)
train_rot_loader = data.DataLoader(train_rot_dataset, batch_size=20)