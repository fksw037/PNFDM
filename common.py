from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_closing, binary_opening
import json
import cv2


REPO_NAME = "facebookresearch/dinov2"
MODEL_NAMES = {"large" : "dinov2_vitg14",
               "medium": "dinov2_vitl14",
               "small" : "dinov2_vitb14",
               "tiny"  : "dinov2_vits14",
               "large_reg" : "dinov2_vitg14_reg",
               "medium_reg": "dinov2_vitl14_reg",
               "small_reg" : "dinov2_vitb14_reg",
               "tiny_reg"  : "dinov2_vits14_reg"}

class DINOv2:
    """
    DINOv2 class uses Meta's DINOv2 model to extract feature embeddings of an image.

    Attributes:
        smaller_edge_size (int): Size of the smaller edge of the image for processing.
        half_precision (bool): Flag to indicate if half precision computation is used.
        device (str): Computing device, e.g., 'cuda' or 'cpu'.
        model (torch.Module): Loaded DINO model.
        transform (torchvision.transforms): Transformations applied to the input image.
    """

    def __init__(self,
                 model_size: str = "small_reg",
                 smaller_edge_size: int = 448,
                 half_precision: bool = False,
                 device: str = "cuda",
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        Initializes the Sparse_Matcher with specified parameters and loads the DINO model.

        Arguments:
            model_size (str, "small"): Size of the DINO model to load in terms of number of params
                'large' = 1.1 B
                'medium' = 300 M
                'small' = 86 M
                'tiny' = 21 M
            smaller_edge_size (int, 448): Size of the smaller edge of the image for processing.
            half_precision (bool, False): Whether to use half precision computation.
            device (str, "cuda"): The computing device, e.g., 'cuda' or 'cpu'.
        """

        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.device = device

        # Loading the DINO model with optional half precision.
        # self.model = torch.hub.load(repo_or_dir=REPO_NAME, model=MODEL_NAMES[model_size])
        self.model = torch.hub.load(repo_or_dir='/root/.cache/torch/hub/facebookresearch_dinov2_main/', source='local',model=MODEL_NAMES[model_size])
        if self.half_precision:
            self.model = self.model.half()  # Convert to half precision if enabled
        self.model = self.model.to(self.device)  # Move model to specified device
        self.model.eval()  # Set model to evaluation mode

        # Transformations for input image preprocessing.
        self.transform = transforms.Compose([
            transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet defaults
            # transforms.Normalize(mean=(0.3399, 0.3399, 0.3399), std=(0.2225, 0.2225, 0.2225)), # Custom-suspension
            transforms.Normalize(mean=mean, std=std)
            ])

    def prepare_image(self, rgb_image_numpy: np.ndarray) -> (torch.Tensor, tuple[int], float):
        """
        Prepares an RGB image for processing by resizing and cropping to fit the model's requirements.

        Arguments:
            rgb_image_numpy (numpy.ndarray): The RGB image in NumPy array format.

        Returns:
            Tuple containing the processed image tensor, grid size, and resize scale.
        """

        # Convert NumPy array to PIL image and apply transformations.
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)
        resize_scale = image.width / image_tensor.shape[2]  # Calculate scale of resize

        # Cropping the image to fit the model's input requirements.
        height, width = image_tensor.shape[1:]  # Extracting height and width
        # Ensure dimensions are multiples of the patch size.
        cropped_width = width - width % self.model.patch_size
        cropped_height = height - height % self.model.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        # Calculate grid size based on the cropped image dimensions.
        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image_tensor, grid_size, resize_scale

    def prepare_mask(self, mask_image_numpy: np.ndarray, grid_size: tuple[int], resize_scale: float) -> np.ndarray:
        """
        Prepares a mask image for processing, aligning it with the dimensions of the processed main image.

        Arguments:
            mask_image_numpy (numpy.ndarray): The mask image in NumPy array format.
            grid_size (tuple[int]): The grid size of the processed main image.
            resize_scale (float): The scale at which the main image was resized.

        Returns:
            NumPy array of the resized mask.
        """

        # Crop and resize mask to align with the processed image's grid.
        cropped_mask_image_numpy = mask_image_numpy[:int(grid_size[0]*self.model.patch_size*resize_scale), :int(grid_size[1]*self.model.patch_size*resize_scale)]
        image = Image.fromarray(cropped_mask_image_numpy)
        resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
        resized_mask = np.asarray(resized_mask).flatten()  # Flatten the resized mask
        return resized_mask

    def extract_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Extracts features from an image tensor using the DINOv2 model.

        Arguments:
            image_tensor (torch.Tensor): The image tensor to extract features from.

        Returns:
            NumPy array of extracted features.
        """


        # Perform inference without gradient calculation for efficiency.
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)

            # Extracting features (tokens) from the image.
            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        return tokens.cpu().numpy()  # Return the extracted features as a NumPy array
    def extract_features_batch(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Extracts features from an image tensor using the DINOv2 model.

        Arguments:
            image_tensor (torch.Tensor): The image tensor to extract features from.

        Returns:
            NumPy array of extracted features.
        """


        # Perform inference without gradient calculation for efficiency.
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.half().to(self.device)
            else:
                image_batch = image_tensor.to(self.device)

            # Extracting features (tokens) from the image.
            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        return tokens.cpu().numpy()  # Return the extracted features as a NumPy array


    def idx_to_source_position(self, idx: int, grid_size: tuple, resize_scale: float) -> (int, int):
        """
        Converts an index in the flattened feature map back to its original position in the source image.

        Arguments:
            idx (int): The index in the flattened feature map.
            grid_size (tuple): The grid size of the processed image.
            resize_scale (float): The scale at which the original image was resized.

        Returns:
            Tuple of row and column indicating the position in the original image.
        """

        # Calculating the row and column in the original image from the index.
        row = (idx // grid_size[1]) * self.model.patch_size * resize_scale + self.model.patch_size / 2
        col = (idx % grid_size[1]) * self.model.patch_size * resize_scale + self.model.patch_size / 2
        return row, col

    def get_embedding_visualization(self, tokens: np.ndarray, grid_size: tuple, resized_mask: np.ndarray = None) -> np.ndarray:
        """
        Generates a visualization of the embeddings using PCA.

        Arguments:
            tokens (numpy.ndarray): The feature tokens extracted from the image.
            grid_size (tuple): The grid size of the processed image.
            resized_mask (numpy.ndarray, optional): The resized mask for selecting specific tokens.

        Returns:
            NumPy array representing the PCA-reduced and normalized tokens for visualization.
        """

        # Applying PCA to reduce the feature dimensions for visualization.
        pca = PCA(n_components=3)
        if resized_mask is not None:
            tokens = tokens[resized_mask]  # Apply mask if provided
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))

        # Reformatting tokens for visualization based on the resized mask.
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))

        # Normalizing tokens for better visualization.
        normalized_tokens = (reduced_tokens - np.min(reduced_tokens)) / (np.max(reduced_tokens) - np.min(reduced_tokens))
        return normalized_tokens
class DINOv2_MvTEC:
    """
    DINOv2 class uses Meta's DINOv2 model to extract feature embeddings of an image.

    Attributes:
        smaller_edge_size (int): Size of the smaller edge of the image for processing.
        half_precision (bool): Flag to indicate if half precision computation is used.
        device (str): Computing device, e.g., 'cuda' or 'cpu'.
        model (torch.Module): Loaded DINO model.
        transform (torchvision.transforms): Transformations applied to the input image.
    """

    def __init__(self,
                 model_size: str = "small_reg",
                 resize: int = 512, 
                 crop_size: int = 448,
                 half_precision: bool = False,
                 device: str = "cuda",
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        Initializes the Sparse_Matcher with specified parameters and loads the DINO model.

        Arguments:
            model_size (str, "small"): Size of the DINO model to load in terms of number of params
                'large' = 1.1 B
                'medium' = 300 M
                'small' = 86 M
                'tiny' = 21 M
            smaller_edge_size (int, 448): Size of the smaller edge of the image for processing.
            half_precision (bool, False): Whether to use half precision computation.
            device (str, "cuda"): The computing device, e.g., 'cuda' or 'cpu'.
        """

        self.resize = resize
        self.crop_size = crop_size
        self.half_precision = half_precision
        self.device = device

        # Loading the DINO model with optional half precision.
        # self.model = torch.hub.load(repo_or_dir=REPO_NAME, model=MODEL_NAMES[model_size])
        self.model = torch.hub.load(repo_or_dir='/root/.cache/torch/hub/facebookresearch_dinov2_main/', source='local',model=MODEL_NAMES[model_size])
        if self.half_precision:
            self.model = self.model.half()  # Convert to half precision if enabled
        self.model = self.model.to(self.device)  # Move model to specified device
        self.model.eval()  # Set model to evaluation mode

        # Transformations for input image preprocessing.
        self.transform = transforms.Compose([
            transforms.Resize(size=resize, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet defaults
            # transforms.Normalize(mean=(0.3399, 0.3399, 0.3399), std=(0.2225, 0.2225, 0.2225)), # Custom-suspension
            transforms.Normalize(mean=mean, std=std)
            ])

    def prepare_image(self, rgb_image_numpy: np.ndarray) -> (torch.Tensor, tuple[int], float):
        """
        Prepares an RGB image for processing by resizing and cropping to fit the model's requirements.

        Arguments:
            rgb_image_numpy (numpy.ndarray): The RGB image in NumPy array format.

        Returns:
            Tuple containing the processed image tensor, grid size, and resize scale.
        """

        # Convert NumPy array to PIL image and apply transformations.
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)
        resize_scale = image.width / image_tensor.shape[2]  # Calculate scale of resize

        # Cropping the image to fit the model's input requirements.
        height, width = image_tensor.shape[1:]  # Extracting height and width
        # Ensure dimensions are multiples of the patch size.
        cropped_width = width - width % self.model.patch_size
        cropped_height = height - height % self.model.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        # Calculate grid size based on the cropped image dimensions.
        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image_tensor, grid_size, resize_scale

    def prepare_mask(self, mask_image_numpy: np.ndarray, grid_size: tuple[int], resize_scale: float) -> np.ndarray:
        """
        Prepares a mask image for processing, aligning it with the dimensions of the processed main image.

        Arguments:
            mask_image_numpy (numpy.ndarray): The mask image in NumPy array format.
            grid_size (tuple[int]): The grid size of the processed main image.
            resize_scale (float): The scale at which the main image was resized.

        Returns:
            NumPy array of the resized mask.
        """

        # Crop and resize mask to align with the processed image's grid.
        cropped_mask_image_numpy = mask_image_numpy[:int(grid_size[0]*self.model.patch_size*resize_scale), :int(grid_size[1]*self.model.patch_size*resize_scale)]
        image = Image.fromarray(cropped_mask_image_numpy)
        resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
        resized_mask = np.asarray(resized_mask).flatten()  # Flatten the resized mask
        return resized_mask

    def extract_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Extracts features from an image tensor using the DINOv2 model.

        Arguments:
            image_tensor (torch.Tensor): The image tensor to extract features from.

        Returns:
            NumPy array of extracted features.
        """


        # Perform inference without gradient calculation for efficiency.
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)

            # Extracting features (tokens) from the image.
            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        return tokens.cpu().numpy()  # Return the extracted features as a NumPy array
    def extract_features_batch(self, image_tensor: torch.Tensor) -> np.ndarray:
            """
            Extracts features from an image tensor using the DINOv2 model.

            Arguments:
                image_tensor (torch.Tensor): The image tensor to extract features from.

            Returns:
                NumPy array of extracted features.
            """


            # Perform inference without gradient calculation for efficiency.
            with torch.inference_mode():
                if self.half_precision:
                    image_batch = image_tensor.half().to(self.device)
                else:
                    image_batch = image_tensor.to(self.device)

                # Extracting features (tokens) from the image.
                tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
            return tokens.cpu().numpy()  # Return the extracted features as a NumPy array



    def idx_to_source_position(self, idx: int, grid_size: tuple, resize_scale: float) -> (int, int):
        """
        Converts an index in the flattened feature map back to its original position in the source image.

        Arguments:
            idx (int): The index in the flattened feature map.
            grid_size (tuple): The grid size of the processed image.
            resize_scale (float): The scale at which the original image was resized.

        Returns:
            Tuple of row and column indicating the position in the original image.
        """

        # Calculating the row and column in the original image from the index.
        row = (idx // grid_size[1]) * self.model.patch_size * resize_scale + self.model.patch_size / 2
        col = (idx % grid_size[1]) * self.model.patch_size * resize_scale + self.model.patch_size / 2
        return row, col

    def get_embedding_visualization(self, tokens: np.ndarray, grid_size: tuple, resized_mask: np.ndarray = None) -> np.ndarray:
        """
        Generates a visualization of the embeddings using PCA.

        Arguments:
            tokens (numpy.ndarray): The feature tokens extracted from the image.
            grid_size (tuple): The grid size of the processed image.
            resized_mask (numpy.ndarray, optional): The resized mask for selecting specific tokens.

        Returns:
            NumPy array representing the PCA-reduced and normalized tokens for visualization.
        """

        # Applying PCA to reduce the feature dimensions for visualization.
        pca = PCA(n_components=3)
        if resized_mask is not None:
            tokens = tokens[resized_mask]  # Apply mask if provided
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))

        # Reformatting tokens for visualization based on the resized mask.
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))

        # Normalizing tokens for better visualization.
        normalized_tokens = (reduced_tokens - np.min(reduced_tokens)) / (np.max(reduced_tokens) - np.min(reduced_tokens))
        return normalized_tokens


class Sparse_Matcher:
    def __init__(self):
        pass
    def extract_features_with_mask(self,features, grid_size, mask):
        """
        提取mask为1位置的特征点的坐标和对应的特征向量.

        :param features: numpy array, 形状为 (num_patches, feature_dim)
        :param grid_size: tuple, (height_patches, width_patches)
        :param mask: numpy array, 形状为 (num_patches,)
        :return: 特征点的坐标和对应的特征向量
        """
        # 获取grid_size
        height_patches, width_patches = grid_size

        # 确保特征向量数量与grid_size一致
        assert features.shape[0] == height_patches * width_patches, "Features数量与grid_size不匹配"
        assert mask.shape[0] == height_patches * width_patches, "Mask数量与grid_size不匹配"

        # 找到mask中值为1的索引
        mask_indices = np.argwhere(mask == 1).flatten()

        # 提取对应的特征向量
        extracted_features = features[mask_indices]

        # 计算特征点的坐标
        coordinates = [(idx // width_patches, idx % width_patches) for idx in mask_indices]

        # 返回特征点的坐标和对应的特征向量
        return coordinates, extracted_features
    
    def match_features(self, features1: np.ndarray, features2: np.ndarray) -> (np.ndarray, np.ndarray):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(features1)
        distances, match2to1 = knn.kneighbors(features2)
        # match2to1 = np.array(match2to1)
        match2to1=np.squeeze(match2to1)
        match2to1 = match2to1.tolist()

        return distances, match2to1
    
    def visualize_costum(self,
                  image1: np.ndarray,
                  image2: np.ndarray,
                  coordinates1: np.ndarray,
                  coordinates2: np.ndarray,
                  distances:np.ndarray, 
                  match2to1:np.ndarray,
                  patch_size: int =14,    
                  figsize: tuple[int] = (20, 20),
                  show_percentage: float = 1,
                  coord_add05:bool =True):

        
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(image1)
        ax1.axis("off")
        ax2.imshow(image2)
        ax2.axis("off")

        for idx2, (dist, idx1) in enumerate(zip(distances, match2to1)):
            row1, col1 = coordinates1[idx1]
            row2, col2 = coordinates2[idx2]
            if coord_add05:
                xyA = (col1 * patch_size, row1 * patch_size)
                xyB = (col2 * patch_size, row2 * patch_size)
            else:
                xyA = ((col1+0.5) * patch_size, (row1+0.5) * patch_size)
                xyB = ((col2+0.5) * patch_size, (row2+0.5) * patch_size)

            if np.random.rand() > show_percentage: continue # sparsely draw so that we can see the lines...

            con = ConnectionPatch(xyA=xyB, xyB=xyA, coordsA="data", coordsB="data",
                                  axesA=ax2, axesB=ax1, color=np.random.rand(3,))
            ax2.add_artist(con)

    def visualize(self,
                  dinov2: DINOv2,
                  image1: np.ndarray,
                  image2: np.ndarray,
                  mask1: np.ndarray,
                  mask2: np.ndarray,
                  grid_size1: tuple[int],
                  grid_size2: tuple[int],
                  resize_scale1: float,
                  resize_scale2: float,
                  distances: np.ndarray,
                  match2to1: np.ndarray,
                  figsize: tuple[int] = (20, 20),
                  show_percentage: float = 1):

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(image1)
        ax1.axis("off")
        ax2.imshow(image2)
        ax2.axis("off")

        for idx2, (dist, idx1) in enumerate(zip(distances, match2to1)):
            row, col = dinov2.idx_to_source_position(idx1, grid_size1, resize_scale1)
            xyA = (col, row)
            if not mask1[int(row), int(col)]: continue # skip if feature is not on the object

            row, col = dinov2.idx_to_source_position(idx2, grid_size2, resize_scale2)
            xyB = (col, row)
            if not mask2[int(row), int(col)]: continue # skip if feature is not on the object

            if np.random.rand() > show_percentage: continue # sparsely draw so that we can see the lines...

            con = ConnectionPatch(xyA=xyB, xyB=xyA, coordsA="data", coordsB="data",
                                    axesA=ax2, axesB=ax1, color=np.random.rand(3,))
            ax2.add_artist(con)



def parse_segmentation_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    result = {
        "add": [],
        "sub": [],
        "imageWidth": data['imageWidth'],
        "imageHeight": data['imageHeight']
    }

    for shape in data['shapes']:
        if shape['label'] == 'add':
            result['add'].append(shape['points'])
        elif shape['label'] == 'sub':
            result['sub'].append(shape['points'])

    return result
def resize_mask(mask, new_size):
    return cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
def scale_points(points, scale_x, scale_y):
    return [[point[0] * scale_x, point[1] * scale_y] for point in points]

def create_binary_mask_resize(scaled_add_points, scaled_sub_points, mask_size):
    mask = np.zeros(mask_size, dtype=np.uint8)

    for points in scaled_add_points:
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 1)

    for points in scaled_sub_points:
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 0)

    return mask

def extract_keypoints_and_descriptors(dense_features):
    H, W, D = dense_features.shape
    keypoints = []
    descriptors = []

    for y in range(H):
        for x in range(W):
            # 每个像素位置作为一个特征点
            kp = cv2.KeyPoint()
            kp.pt = (x, y)
            
            keypoints.append(kp)
            # 提取该像素位置的特征向量作为描述子
            descriptors.append(dense_features[y, x, :])

    # 转换描述子为 numpy 数组
    descriptors = np.array(descriptors, dtype=np.float32)

    return keypoints, descriptors

def Matches_Horizon(kpt_src, kpt_dst, desc_src, desc_dst, SearchRegion=(-100, 100), thUspDist=2, r=5):
    '''
    *两帧图像稀疏立体匹配（即：USP特征点匹配，非逐像素的密集匹配，但依然满足行对齐）
     * 输入：两帧立体矫正后的图像img_left 和 img_right 对应的USP特征点集
     * 过程：
          1. 行特征点统计. 统计img_right每一行上的USP特征点集，便于使用立体匹配思路(行搜索/极线搜索）进行同名点搜索, 避免逐像素的判断.
          2. 粗匹配. 根据步骤1的结果，对img_left第i行的USP特征点pi，在img_right的第i行上的USP特征点集中搜索相似USP特征点, 得到qi
          3. 精确匹配. 以点qi为中心，半径为r的范围内，进行块匹配（归一化SAD），进一步优化匹配结果
          4. 亚像素精度优化. 步骤3得到的视差为uchar/int类型精度，并不一定是真实视差，通过亚像素差值（抛物线插值)获取float精度的真实视差
          5. 最优视差值/深度选择. 通过胜者为王算法（WTA）获取最佳匹配点。
          6. 删除离群点(outliers). 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是正确匹配，比如光照变化、弱纹理等会造成误匹配
     * 输出：稀疏特征点视差图/深度图（亚像素精度）mvDepth 匹配结果 mvuRight

    '''
    minD, maxD = SearchRegion
    matches_temp = []  # 创建匹配变量
    # 将特征点坐标转换为NumPy数组
    pts_src = cv2.KeyPoint_convert(kpt_src)
    pts_dst = cv2.KeyPoint_convert(kpt_dst)
    pts_src_mean = pts_src[:, 0].mean()

    pts_dst_mean = pts_dst[:, 0].mean()
    offset_x = pts_dst_mean - pts_src_mean

    # int matcher_TH_HIGH=50;
    # int matcher_TH_LOW=7;
    # // USP特征相似度阈值  -> mean ～= (max  + min) / 2
    # 二维vector存储每一行的usp特征点的列坐标的索引，为什么是vector，因为每一行的特征点有可能不一样，例如
    # vRowIndices[0] = [1，2，5，8, 11]   第1行有5个特征点,他们的列号（即x坐标）分别是1,2,5,8,11
    # vRowIndices[1] = [2，6，7，9, 13, 17, 20]  第2行有7个特征点.etc
    RowIndices = [[] for _ in range(int(pts_src[:, 1].max() + 1))]
    # // 右图特征点数量，N表示数量 r表示右图，且不能被修改

    for iR in range(len(kpt_dst)):

        kpY = int(pts_dst[iR][1])

        # 计算特征点ir在行方向上，可能的偏移范围r，即可能的行号为[kpY + r, kpY - r]

        # 将特征点ir保证在可能的行号中
        maxr = min(kpY + r, len(RowIndices) - 1)
        minr = max(0, kpY - r)
        for yi in range(minr, maxr):
            RowIndices[yi].append(iR)

    # 为左图每一个特征点il，在右图搜索最相似的特征点ir
    for iL in range(len(kpt_src)):
        pt_x = int(pts_src[iL][0])
        pt_y = int(pts_src[iL][1])

        # 获取左图特征点il所在行，以及在右图对应行中可能的匹配点
        Candidates = RowIndices[pt_y]
        if (not len(Candidates)): continue

        # 计算理论上的最佳搜索范围
        maxU = min(int(pts_dst[:, 0].max()), pt_x + offset_x + maxD)
        minU = max(int(pts_dst[:, 0].min()), int(pt_x + offset_x + minD))
        # 初始化最佳相似度，用最大相似度，以及最佳匹配点索引
        bestDist = np.inf
        bestIdxR = -1
        dL = desc_src[iL]
        # 粗配准。左图特征点il与右图中的可能的匹配点进行逐个比较, 得到最相似匹配点的描述子距离和索引
        for iC in range(len(Candidates)):
            iR = Candidates[iC]
            uR = pts_dst[iR][0]
            # 超出理论搜索范围[minU, maxU]，可能是误匹配，放弃
            if (uR >= minU and uR <= maxU):
                # 计算匹配点il和待匹配点ic的相似度dist
                dR = desc_dst[iR]
                dist = np.linalg.norm(dL - dR)
                # 统计最小相似度及其对应的列坐标(x)
                if (dist < bestDist):
                    bestDist = dist
                    bestIdxR = iR
        if (bestDist < thUspDist):
            # 如果刚才匹配过程中的最佳描述子距离小于给定的阈值
            # 计算右图特征点x坐标
            # uR0 = kpt_right[bestIdxR].pt.x;
            match = cv2.DMatch()
            match.distance = bestDist
            match.imgIdx = 0  # 0-表示原有匹配，1-表示通过变换新获得的匹配
            match.queryIdx = iL
            match.trainIdx = bestIdxR
            matches_temp.append(match)
    return tuple(matches_temp)

def polynomial_fit_ransac(x_org, y_org, threshold=0.1, max_iterations=100, sample_size=3):

    best_model = None
    best_inliers = None
    best_inlier_count = 0
    x=x_org[:,0]
    y=y_org[:,0]
    for _ in range(max_iterations):
        # Randomly select a subset of points
        sample_indices = np.random.choice(x.size, size=sample_size)
        sample_x = x[sample_indices]
        sample_y = y[sample_indices]

        # Fit a polynomial model using the selected points
        model = np.polyfit(sample_x, sample_y, deg=3)
        predicted_y = np.polyval(model, x)

        # Calculate distances between points and the curve
        distances=np.abs(y-predicted_y)

        # Identify inliers and outliers based on the threshold
        inlier_indices = np.where(distances < threshold)
        outlier_indices = np.where(distances >= threshold)

        inlier_count = inlier_indices[0].size

        # Check if this model has more inliers than the previous best model
        if inlier_count > best_inlier_count:
            best_model = model
            best_inliers = inlier_indices[0]
            best_inlier_count = inlier_count

    return best_model, best_inliers
def ConvertNumpy2CVKpt(scr_pt):
    Kpts = []
    for elemt in scr_pt:
        x, y= elemt.ravel()
        point_temp = cv2.KeyPoint()
        point_temp.pt = (x, y)
        Kpts.append(point_temp)
    Kpts = tuple(Kpts)
    return Kpts
def PolynomialFilterSingleTemplateAllPoints(Kpts1_choose,Kpts2_choose,matches_temp,distance_the=5.0):
    '''
    去除侧面字匹配的影响
    '''
    # Extract matching keypoints from each image
    # Extract matching keypoints from each image
    src_pts = []
    dst_pts = []
    status_record = []
    index_record = []
    pts_id = 0
    if len(matches_temp):
        for m in matches_temp:
            src_pts.append(Kpts1_choose[m.queryIdx].pt)
            dst_pts.append(Kpts2_choose[m.trainIdx].pt)
            status_record.append(m.imgIdx)
            index_record.append(pts_id)
            pts_id = pts_id + 1
    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    # Compute affine transformation
    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, distance_the)
    # Fit a polynomial using RANSAC
    M, mask = polynomial_fit_ransac(src_pts, dst_pts, threshold=distance_the, max_iterations=100,
                                             sample_size=3)
    src_pts_choose = []
    dst_pts_choose = []
    match_status_choose = []
    for i in mask:
        if i in index_record:
            src_pts_choose.append(src_pts[i])
            dst_pts_choose.append(dst_pts[i])
            match_status_choose.append(status_record[index_record.index(i)])
    Kpts1_cv = []
    Kpts2_cv = []
    matches = []
    if len(src_pts_choose):
        Kpts1_cv = ConvertNumpy2CVKpt(src_pts_choose)
        Kpts2_cv = ConvertNumpy2CVKpt(dst_pts_choose)

    for i in range(len(Kpts1_cv)):
        match = cv2.DMatch()
        match.distance = 1
        match.imgIdx = match_status_choose[i]
        match.queryIdx = i
        match.trainIdx = i
        matches.append(match)
    matches = tuple(matches)

    return Kpts1_cv, Kpts2_cv, matches, M
from enum import Enum
class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    COLOR_CODED_POINTS_X = 3
    COLOR_CODED_POINTS_Y = 4
    COLOR_CODED_POINTS_XpY = 5
def draw_matches_vertical(src1, src2, kp1, kp2, matches, drawing_type=DrawingType.LINES_AND_POINTS):
    height = src1.shape[0] + src2.shape[0]
    width =  max(src1.shape[1], src2.shape[1])
    if len(src1.shape)>2:
        output = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        output = np.zeros((height, width), dtype=np.uint8)

    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[src1.shape[0]:,0:src2.shape[1]] = src2[:]
    
    if drawing_type == DrawingType.ONLY_LINES:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (0,src1.shape[0])))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

    elif drawing_type == DrawingType.LINES_AND_POINTS:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (0,src1.shape[0])))
            if(matches[i].imgIdx==1):
                cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 0, 255))
            else:
                cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (0,src1.shape[0])))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (0,src1.shape[0])))

            if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                colormap_idx = int((left[0] - src1.shape[1]*.5 + left[1] - src1.shape[0]*.5) * 256. / (src1.shape[0]*.5 + src1.shape[1]*.5))  # manhattan gradient

            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv2.circle(output, tuple(map(int, left)), 1, color, 2)
            cv2.circle(output, tuple(map(int, right)), 1, color, 2)
    return output

def create_binary_mask(result):
    mask = np.zeros((result['imageHeight'], result['imageWidth']), dtype=np.uint8)

    for points in result['add']:
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 1)

    for points in result['sub']:
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 0)

    return mask

def filter_keypoints_by_mask(keypoints, descriptors, mask):
    filtered_keypoints = []
    filtered_descriptors = []

    for kp, desc in zip(keypoints, descriptors):
        x, y = kp.pt
        x = int(x)
        y = int(y)
        if mask[y, x] == 1:
            filtered_keypoints.append(kp)
            filtered_descriptors.append(desc)

    return filtered_keypoints, np.array(filtered_descriptors, dtype=np.float32)
