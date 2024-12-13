from common import DINOv2
#多图PCA 特征可视化
import torch
import os
import  cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import time
from torch.utils.data import DataLoader,Dataset
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import mahalanobis
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from torchvision import transforms as T
import pandas as pd
from scipy.stats import chi2
import seaborn as sns

class CustomDataset(Dataset):
    def __init__(self, root_path='../data',resize=448,mean=(0.2451, 0.2451, 0.2451), std=(0.1462, 0.1462, 0.1462)):
        
        self.root_path = root_path
        self.resize = resize
        # load dataset
        self.x = self.load_dataset_folder()
        # set transforms
        self.transform_x = T.Compose([T.Resize(size=resize, interpolation=T.InterpolationMode.BICUBIC, antialias=True),                                      
                                      T.ToTensor(),
                                      T.Normalize(mean, std)])

    def __getitem__(self, idx):
        x=self.x[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        return x,os.path.basename(self.x[idx])

    def __len__(self):
        return len(self.x)
    def load_dataset_folder(self):
        x=[]
        img_dir = self.root_path

        # load images
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                    for f in os.listdir(img_dir)
                                    if f.endswith('.png')])
        x.extend(img_fpath_list)
        return list(x)



# 定义一个结构体变量GMM
class GMM:
    def __init__(self, n_components, means, covariances):
        self.n_components = n_components
        self.means_ = means
        self.covariances_ = covariances
def get_neighbors(data, index, r, h, w):
    neighbors = []
    i, j = divmod(index, w)
    num_images, _, _ = data.shape

    positions = []
    for x in range(max(0, i - r), min(h, i + r + 1)):
        for y in range(max(0, j - r), min(w, j + r + 1)):
            neighbors.append(data[:, x * w + y, :])
            positions.append((x, y))

    neighbors = np.array(neighbors).reshape(-1, data.shape[-1])
    
    # # 绘制用于GMM拟合的向量位置
    # plt.figure()
    # plt.title(f"Position {i}, {j} with radius {r}")
    # plt.scatter(*zip(*positions), c='red')
    # plt.xlim(0, h)
    # plt.ylim(0, w)
    # plt.gca().invert_yaxis()
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    
    return neighbors


def process_feature(index, data, r, h, w):
    neighbors = get_neighbors(data, index, r, h, w)
    lowest_aic = np.inf
    best_n_components = 1
    if N_COMPONENTS!=1:
        n_components_range = range(1, 10)

        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(neighbors)
            aic_score = gmm.aic(neighbors)
            if aic_score < lowest_aic:
                lowest_aic = aic_score
                best_n_components = n_components
    # print(f"Best_n_components for index {index}: {best_n_components}")
    best_gmm = GaussianMixture(n_components=best_n_components, random_state=42)
    best_gmm.fit(neighbors)
    best_gmm= GMM(best_gmm.n_components, best_gmm.means_, best_gmm.covariances_)
    return best_gmm, best_n_components




if __name__ == "__main__":

    r=2
    N_COMPONENTS=1#自动计算最佳的组件数量设置为-1
    PCA_COMPONENTS=512#PCA降维维度设置
    ImgResize=448
    # 定义一个非常小的概率值，接近于0
    epsilon_chi2 = 5e-6#决定卡方分布边界值的阈值(调节热图信噪比)
    chi2_boundary_value_manmul=None#30 #训练训练图片较少时，全要设定分割阈值
    segment_threshold=0.5#热图分割阈值
    datasetmean=(0.2451, 0.2451, 0.2451)#统计数据集图像分布
    datasetstd=(0.1462, 0.1462, 0.1462)
   
    Dataset='RoofAD-Hard-SubRand1'
    
    DirDict={  
            'ImageTrainDir': 'DataSet/region-2/train/good',
            'ImageTestDir':'DataSet/region-2/test/foreign_objects', 
            'GMMModelDir':f'GMM-Models-PCA/{Dataset}',
            'PCA_ModelDir':f'PCA-Models/{Dataset}',
            'TrainDataDistribDir':f'Train-Data-Distrib/{Dataset}',            
            'ResultSaveDir':f'CustomResult/{Dataset}',      
            }


    # 设置设备为GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ImageTrainDir =DirDict['ImageTrainDir']
    ImageTestDir =DirDict['ImageTestDir']

    GMMModelDir=DirDict['GMMModelDir']
    PCA_ModelDir=DirDict['PCA_ModelDir']
    TrainDataDistribDir= DirDict['TrainDataDistribDir']
    ResultSaveDir=DirDict['ResultSaveDir']

    os.makedirs(ResultSaveDir, exist_ok=True)
    os.makedirs(GMMModelDir, exist_ok=True)
    gmm_model_name =f'gmm_list-r{r}-n{N_COMPONENTS}-pca-{PCA_COMPONENTS}.pkl'
    gmm_model_path = os.path.join(GMMModelDir, gmm_model_name)

    os.makedirs(PCA_ModelDir, exist_ok=True)
    pca_model_name=f'r{r}-n{N_COMPONENTS}-pca-{PCA_COMPONENTS}.pkl'
    pca_model_path = os.path.join(PCA_ModelDir, pca_model_name)    


    
    os.makedirs(TrainDataDistribDir, exist_ok=True)
    train_data_distrib_name=f'pca-{PCA_COMPONENTS}-r{r}-n{N_COMPONENTS}-train_data_distrib.csv'
    train_data_distrib_path = os.path.join(TrainDataDistribDir, train_data_distrib_name)
    
    train_dataset = CustomDataset(ImageTrainDir, resize=ImgResize,mean=datasetmean, std=datasetstd)
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)

    dinov2 = DINOv2(model_size= "small_reg",smaller_edge_size=ImgResize,mean=datasetmean, std=datasetstd)    

    embed_dim=dinov2.model.embed_dim
    grid_size=(ImgResize//14,ImgResize//14)    
    
    
    gmm_models=[] 
    pca_models=[]
    chi2_boundary_value=[]

    if os.path.exists(gmm_model_path) and os.path.exists(pca_model_path) and os.path.exists(train_data_distrib_path):
        with open(gmm_model_path, 'rb') as f:
            gmm_models = pickle.load(f)
            print(f'GMM model loaded from {gmm_model_path}')
        with open(pca_model_path, 'rb') as f:
            pca_models = pickle.load(f)
            print(f'PCA model loaded from {pca_model_path}')
        data_train = pd.read_csv(train_data_distrib_path).values.flatten()
        print(f'Train data distribution loaded from {train_data_distrib_path}')
        # 对平方后的数据进行卡方分布拟合
        df_estimated, loc, scale = chi2.fit(data_train, floc=13.5)
        # 定义一个非常小的概率值，接近于0
        # epsilon = 1e-10
        # epsilon = 1e-11
        # 通过逆累积分布函数找到对应的边界值
        chi2_boundary_value = chi2.ppf(1 - epsilon_chi2, df_estimated, loc, scale)
        print(f'chi2_boundary_value | Test Phase:{chi2_boundary_value}')
       #===绘制卡方分布=====
        # 绘制卡方分布的PDF
        x = np.linspace(0, chi2_boundary_value, 1000)
        pdf_fitted = chi2.pdf(x, df_estimated, loc, scale)

        plt.figure(figsize=(10, 6))
        sns.histplot(data_train, bins=200, kde=False, stat='density', color='skyblue', label='Data Histogram')

        plt.plot(x, pdf_fitted, 'r-', lw=2, label=f'Chi-Square Distribution (df={df_estimated}, loc={loc}, scale={scale})')
        plt.axvline(chi2_boundary_value, color='g', linestyle='--', label=f'Boundary Value: {chi2_boundary_value:.2f}')

        # 添加图例和标签
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Chi-Square Distribution and Boundary Value')
        plt.legend()
        plt.grid(True)
        os.makedirs(os.path.join(ResultSaveDir,f'Draw_PCA-{PCA_COMPONENTS}-ACCE/r{r}-n{N_COMPONENTS}'),exist_ok=True)
        plt.savefig(os.path.join(ResultSaveDir,f'Draw_PCA-{PCA_COMPONENTS}-ACCE/r{r}-n{N_COMPONENTS}','Chi-Square Distribution and Boundary Value.png'))

        #===开始测试==
    else:
        #===开始训练===
        print(f'GMM model not found in {gmm_model_path}')
        print(f'PCA model not found in {pca_model_path}')
        print(f'Train data distribution not found in {train_data_distrib_path}')

        total_features_train  = []   
        with torch.no_grad():
            for (imgs,img_names) in tqdm(train_dataloader, 'Feature extraction | Training | %s |' % Dataset):
                
                
                # Extract image1 features
                features = dinov2.extract_features_batch(imgs)
                total_features_train.append(features)

        total_features_train = np.concatenate(total_features_train, axis=0)
        # PCA降维处理            
        pca_models = PCA(n_components=PCA_COMPONENTS)
        patch_features_pca = total_features_train.reshape(-1, total_features_train.shape[-1])
        patch_features_pca = pca_models.fit_transform(patch_features_pca)
        patch_features_pca=patch_features_pca.reshape(total_features_train.shape[0],-1, PCA_COMPONENTS)

        # 保存PCA模型
        with open(pca_model_path, 'wb') as f:
            pickle.dump(pca_models, f)
        print(f'PCA model has been saved in {pca_model_path}')
        h, w = grid_size

        # 使用joblib并行处理
        num_jobs = -1  # 使用所有可用的CPU核心
        num_jobs = 92  # 使用所有可用的CPU核心
        results = Parallel(n_jobs=num_jobs, backend="loky")(
            delayed(process_feature)(i, patch_features_pca, r, h, w) for i in tqdm(range(h * w), desc=f"Fit GMM Models")
        )

        gmm_models, best_n_list = zip(*results)

        # 保存 GMM 列表到文件
        with open(gmm_model_path, 'wb') as f:
            pickle.dump(gmm_models, f)
        print(f'Gmm model has been saved in {gmm_model_path}')


        #======统计训练集的马氏距离分布=====
        mahal_distances_total=[]
        gmm_mu=[]
        gmm_simga=[]
        for gmm_elemt in gmm_models:
            gmm_mu.append(gmm_elemt.means_,)
            gmm_simga.append(gmm_elemt.covariances_)
        gmm_mu=np.concatenate(gmm_mu, axis=0)
        gmm_simga=np.concatenate(gmm_simga, axis=0) 
        
        
        for features_pca in tqdm(patch_features_pca, 'Statistical Analysis in Progress | Training | %s |' % Dataset):

            epsilon =1e-5
            # 正则化Sigma
            Sigma_reg = torch.from_numpy(gmm_simga).to(device) + epsilon * torch.eye(gmm_simga.shape[1], device=device).unsqueeze(0)

            # 批量计算差值
            diff = (torch.from_numpy(features_pca).to(device) - torch.from_numpy(gmm_mu).to(device)).unsqueeze(-1)
            scale=1
            sol = torch.linalg.solve(Sigma_reg, diff)*scale
            mahalanobis_dist_sq = torch.matmul(diff.transpose(1, 2), sol).squeeze()
            # 取平方根得到马氏距离
            mahalanobis_dist = torch.sqrt(mahalanobis_dist_sq)
            mahal_distances=mahalanobis_dist.view(grid_size).cpu().numpy()
            mahal_distances_total.append(mahal_distances.reshape(-1))

        mahal_distances_total = np.stack(mahal_distances_total, axis=0)

        distance_samples=mahal_distances_total.flatten()
        # 将NumPy数组保存成CSV文件
        np.savetxt(train_data_distrib_path, distance_samples, delimiter=',')
                # 对平方后的数据进行卡方分布拟合
        df_estimated, loc, scale = chi2.fit(distance_samples, floc=13.5)
        # 定义一个非常小的概率值，接近于0
        # epsilon_chi2 = 1e-10
        # epsilon_chi2 = 1e-11
        # 通过逆累积分布函数找到对应的边界值
        chi2_boundary_value = chi2.ppf(1 - epsilon_chi2, df_estimated, loc, scale)   
        print(f'chi2_boundary_value | Train Phase:{chi2_boundary_value}')   
        #===绘制卡方分布=====
        # 绘制卡方分布的PDF
        x = np.linspace(0, chi2_boundary_value, 1000)
        pdf_fitted = chi2.pdf(x, df_estimated, loc, scale)

        plt.figure(figsize=(10, 6))
        sns.histplot(distance_samples, bins=200, kde=False, stat='density', color='skyblue', label='Data Histogram')

        plt.plot(x, pdf_fitted, 'r-', lw=2, label=f'Chi-Square Distribution (df={df_estimated}, loc={loc}, scale={scale})')
        plt.axvline(chi2_boundary_value, color='g', linestyle='--', label=f'Boundary Value: {chi2_boundary_value:.2f}')

        # 添加图例和标签
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Chi-Square Distribution and Boundary Value')
        plt.legend()
        plt.grid(True)
        os.makedirs(os.path.join(ResultSaveDir,f'Draw_PCA-{PCA_COMPONENTS}-ACCE/r{r}-n{N_COMPONENTS}'),exist_ok=True)
        plt.savefig(os.path.join(ResultSaveDir,f'Draw_PCA-{PCA_COMPONENTS}-ACCE/r{r}-n{N_COMPONENTS}','Chi-Square Distribution and Boundary Value.png'))
        # plt.show()
        plt.close()
    #======开始测试======
        

    
    ResultDir=os.path.join(ResultSaveDir,f'Draw_PCA-{PCA_COMPONENTS}-ACCE/r{r}-n{N_COMPONENTS}')
    os.makedirs(ResultDir,exist_ok=True)

    # 获取文件夹中所有图像文件名并排序
    image_files = [f for f in sorted(os.listdir(ImageTestDir)) if f.endswith('.jpg') or f.endswith('.png')]


    embed_dim=dinov2.model.embed_dim
    gmm_mu=[]
    gmm_simga=[]
    for gmm_elemt in gmm_models:
        gmm_mu.append(gmm_elemt.means_,)
        gmm_simga.append(gmm_elemt.covariances_)
    gmm_mu=np.concatenate(gmm_mu, axis=0)
    gmm_simga=np.concatenate(gmm_simga, axis=0) 

    with torch.no_grad():

        #提取所有图像特征
        time_list=[]
        for file in tqdm(image_files, desc=f"Extracting features | Testing"):
            t_s=time.time()
            image_path = os.path.join(ImageTestDir, file)
            img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            # Extract image1 features
            image_tensor, grid_size, resize_scale = dinov2.prepare_image(img)
            features = dinov2.extract_features(image_tensor)
            features_pca = pca_models.transform(features)
            target_size=grid_size#(32,32)
            h, w = grid_size
            
            epsilon =1e-5
            # 正则化Sigma
            Sigma_reg = torch.from_numpy(gmm_simga).to(device) + epsilon * torch.eye(gmm_simga.shape[1], device=device).unsqueeze(0)

            # 批量计算差值
            diff = (torch.from_numpy(features_pca).to(device) - torch.from_numpy(gmm_mu).to(device)).unsqueeze(-1)
            scale=1
            sol = torch.linalg.solve(Sigma_reg, diff)*scale
            mahalanobis_dist_sq = torch.matmul(diff.transpose(1, 2), sol).squeeze()
            # 取平方根得到马氏距离
            mahalanobis_dist = torch.sqrt(mahalanobis_dist_sq)
            mahal_distances=mahalanobis_dist.view(target_size).cpu().numpy()
            #标准化处理
            mahal_distances_org = mahal_distances.reshape(-1)
            # chi2_boundary_value=35
            if chi2_boundary_value_manmul:
                chi2_boundary_value=chi2_boundary_value_manmul
            mahal_distances_chi2norm = (mahal_distances_org-chi2_boundary_value)/(chi2_boundary_value)
            mahal_distances_chi2norm = mahal_distances_chi2norm.reshape(target_size)
            mahal_distances_chi2norm = np.clip(mahal_distances_chi2norm,0,1)

            mahal_distances_org = mahal_distances_org.reshape(target_size)
            mahal_distances_org_upscale = cv2.resize(mahal_distances_org, (ImgResize,ImgResize), interpolation=cv2.INTER_LINEAR)
       
            mahal_distances_chi2norm_upscale = cv2.resize(mahal_distances_chi2norm, (ImgResize,ImgResize), interpolation=cv2.INTER_LINEAR)
       
            # 创建绘图对象，并设置3个子图
            fig, axs = plt.subplots(1, 4, figsize=(24, 5))

            # 显示原始图像
            axs[0].imshow(img, cmap='gray')
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            # 显示热图
            cax = axs[1].imshow(mahal_distances_org_upscale, cmap='hot')
            axs[1].set_title('HeatmapOrg')
            axs[1].axis('off')
            # 添加颜色条到热图
            cbar = fig.colorbar(cax, ax=axs[1], fraction=0.046, pad=0.04)
            cbar.set_label('Value')

            # 显示热图
            cax = axs[2].imshow(mahal_distances_chi2norm_upscale, cmap='hot')
            axs[2].set_title('HeatmapChi2Norm')
            axs[2].axis('off')
            # 添加颜色条到热图
            cbar = fig.colorbar(cax, ax=axs[2], fraction=0.046, pad=0.04)
            cbar.set_label('Value')  
            # 设置颜色条的范围
            cax.set_clim(0, 1)           
             
            # 计算分割区域的掩膜和边界
            mask_seg = np.where(mahal_distances_chi2norm_upscale > segment_threshold, 1, 0).astype(np.uint8)
            # 找到mask中值为1的区域的轮廓
            image_height, image_width, _ = img.shape            
            # 在原始图像上绘制轮廓
            # 使用红色 (BGR = 0, 0, 255) 绘制，线宽为4
            img_seg=img.copy()
            contours, _ = cv2.findContours(mask_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            scaled_contours = []
            for contour in contours:
                scaled_contour = contour.astype(np.float32)
                scaled_contour[:, 0, 0] = scaled_contour[:, 0, 0] * (image_width / ImgResize)
                scaled_contour[:, 0, 1] = scaled_contour[:, 0, 1] * (image_height / ImgResize)
                scaled_contours.append(scaled_contour.astype(np.int32))
            # # 在原始图像上绘制轮廓
            # cv2.drawContours(img_seg, scaled_contours, -1, (0, 255, 0), 4)
            # 在原始图像上绘制轮廓
            cv2.drawContours(img_seg, scaled_contours, -1, (0, 255, 0), 2)

            # 显示分割后结果并绘制到原图上
            axs[3].imshow(img_seg)
            axs[3].set_title('Segment Result')
            axs[3].axis('off')


            # 显示图像
            plt.tight_layout()
            # plt.show()
            save_result_path=os.path.join(ResultDir,file)
            plt.savefig(save_result_path)
            plt.close()



