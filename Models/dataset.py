class CreateDataset(data.Dataset):
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

def normalize(label):
    for i,j in enumerate(label):
        label[i] = (j - 1.5) / (1.65 - 1.5)
    return label
def denormalize(label):
    label = label * (1.65 - 1.5) + 1.5
    return label