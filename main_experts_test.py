import shutil 
import os 
import numpy as np
import cv2 
import torch
import torch.utils.data as data

import experts.raft_of_expert
import experts.liteflownet_of_expert

import utils.visualize_of

class MultiDomainDataset(data.Dataset):
    def __init__(self, videos_path):
        super(MultiDomainDataset, self).__init__()

        # RAFT OF 
        #self.raft_of = raft_of_test.RaftTest('experts/raft_optical_flow/models/raft-things.pth')
        self.raft_of = experts.raft_of_expert.RaftTest('experts/raft_optical_flow/models/raft-kitti.pth')

         # LiteFlowNet OF 
        self.lite_of = experts.liteflownet_of_expert.LiteFlowNetTest('experts/liteflownet_optical_flow/models/liteflownet-default')

       
        self.samples = []
        video_names = os.listdir(videos_path)
        video_names.sort()
        for video_name in video_names:
            vid_path = os.path.join(videos_path, video_name)
            img_names = os.listdir(vid_path)
            img_names.sort()
            for img_idx in range(len(img_names)):
                indexes = np.arange(img_idx, img_idx+2, 1)
                indexes[indexes >= len(img_names)] = len(img_names)-1
                local_paths = []
                for idx in indexes:
                    local_paths.append(os.path.join(vid_path, img_names[idx]))
                self.samples.append(local_paths)
        
    def __getitem__(self, index):
        
        paths = self.samples[index]

        frame_1 = cv2.imread(paths[0])
        frame_2 = cv2.imread(paths[1])
       
        frame_1_rgb = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
        frame_2_rgb = cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB)

        raft_of = self.raft_of.apply(frame_1_rgb, frame_2_rgb)
        lite_of = self.lite_of.apply(frame_1_rgb, frame_2_rgb)

        frame_1 = torch.from_numpy(frame_1).permute(2, 0, 1).float().cuda()

        return frame_1, raft_of, lite_of

    def __len__(self):
        return len(self.samples)    

def get_sample(img, raft_of, lite_of):
    img = img.permute(1,2,0).numpy()
    raft_of = raft_of.permute(1,2,0).numpy()
    raft_of = utils.visualize_of.flow_to_image(raft_of, clip_flow=None, convert_to_bgr=True)
    lite_of = lite_of.permute(1,2,0).numpy()
    lite_of = utils.visualize_of.flow_to_image(lite_of, clip_flow=None, convert_to_bgr=True)
    img = np.concatenate((img, raft_of, lite_of), 1)
    return img

def save_batch(imgs, raft_ofs, lite_ofs, idx, out_path):

    for i in range(imgs.shape[0]):
        img = get_sample(imgs[i].cpu(), raft_ofs[i].cpu(), lite_ofs[i].cpu())
        if i==0:
            batch_img = img
        else:
            batch_img = np.concatenate((batch_img, img), 0)

    cv2.imwrite(os.path.join(out_path, 'batch_%d.png'%idx), np.uint8(batch_img))

if __name__=="__main__":
    videos_path = r'/root/test_videos'

    dataset = MultiDomainDataset(videos_path)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, drop_last=True)

    out_path = 'out'
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    idx = 0
    for imgs, raft_ofs, lite_ofs in data_loader:   
        
        save_batch(imgs, raft_ofs, lite_ofs, idx, out_path)
        idx = idx+1


