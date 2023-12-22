import cv2
import numpy as np
from typing import Callable, Tuple
from PIL import Image
import time


###################################################################################
                            # SD的环境可以运行
                            # 时间复杂度基本没有变
                            # 主要耗时在加载OpenposeDetector  preprocessor
###################################################################################


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()


def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad



class OpenposeModel(object):
    def __init__(self) -> None:
        self.model_openpose = None
        self.enable_Adtailer = None
        self.mask_num=0
        self.openpose_pic = None
        if self.model_openpose is None:
            from openpose import OpenposeDetector
            self.model_openpose = OpenposeDetector()
    def run_model(
            self,
            img: np.ndarray,
            include_body: bool=True,
            include_hand: bool=False,
            include_face: bool=True,
            json_pose_callback: Callable[[str], None] = None,
            res: int = 512,
            **kwargs  # Ignore rest of kwargs
    ) -> Tuple[np.ndarray, bool]:
        """Run the openpose model. Returns a tuple of
        - result image
        - is_image flag

        The JSON format pose string is passed to `json_pose_callback`.
        """
        print("调用OpenposeModel!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-------------------------------------------------------")
        if json_pose_callback is None:
            json_pose_callback = lambda x: None

        img, remove_pad = resize_image_with_pad(img, res)

        # if self.model_openpose is None:
        #     from openpose import OpenposeDetector
        #     self.model_openpose = OpenposeDetector()
        ########################
        model_openpose = self.model_openpose(
            img,
            include_body=include_body,
            include_hand=include_hand,
            include_face=include_face,
            json_pose_callback=json_pose_callback
        )
        #############################
        # self.mask_num=0
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        people_res=self.model_openpose.results
        print('people_res',people_res)
        face_area = self.model_openpose.calculate_face_area()
        '''
        判断条件：
        1.图里没有人则不启用ADtailer
        2.图里只有一个人，人脸关键点=70且人脸面积>0.1才不修复
        3.图里有多个人，此时打开ADtailer修复人脸
            每张人脸关键点=70且人脸面积>0.05,统计个数
        '''
        if people_res['people_num']==0:
            self.enable_Adtailer=False
        elif people_res['people_num']==1 and people_res['face_points']==70 and face_area>0.1:
            self.enable_Adtailer=False
        else:
            self.enable_Adtailer=True
            # 符合要求,不用修改的人脸数cnt
            cnt=0
            for i in range(len(people_res['area'])):
                if people_res['area'][i]==0:
                    continue
                # 人脸面积>0.05且人脸关键点数量=70
                if people_res['area'][i]>0.05 and people_res['face_points'][i]==70:
                    cnt += 1

            
                self.mask_num=people_res['people_num']-cnt
        if self.mask_num==0 and self.enable_Adtailer==True:
            self.enable_Adtailer=False
        self.openpose_pic = remove_pad(model_openpose)
        # print(self.get_enable_Adtailer_variable)
        self.model_openpose.initialize_attribute()
        # return remove_pad(model_openpose), True
        return self.openpose_pic, True
    
    @property
    def get_enable_Adtailer_variable(self):
        return self.enable_Adtailer,self.mask_num,self.openpose_pic

g_openpose_model = OpenposeModel()
def call(image_path):
    # 打开图片
    # image_path = '/root/openpose_test/1.png'
    image = Image.open(image_path)
    image_np = np.array(image)
    # print(image_np)

    start_time = time.time()

    # 运行获得openpose图与计算结果
    g_openpose_model.run_model(image_np)
    
    # print(g_openpose_model.get_enable_Adtailer_variable)
    end_time = time.time()
    # 计算执行时间
    execution_time = end_time - start_time
    print('获得openpose图与计算结果花费时间：',execution_time)
    return g_openpose_model.get_enable_Adtailer_variable



image_path = '../openpose_test/1.png'
enable_ADtailer,mask_num,openpose_pic=call(image_path)
print('enable_ADtailer:',enable_ADtailer)
print('mask_num:',mask_num)
print('openpose_pic:',openpose_pic)
    
    
# # 卸载模型(存在问题)
# OpenposeModel.unload_model()

# class OpenposeModel_original(object):
#     def __init__(self) -> None:
#         self.model_openpose = None
#         self.enable_Adtailer = None
#         self.mask_num=0
#         self.openpose_pic = None
#         if self.model_openpose is None:
#             from openpose import OpenposeDetector_original
#             self.model_openpose = OpenposeDetector_original()
#     def run_model(
#             self,
#             img: np.ndarray,
#             include_body: bool=True,
#             include_hand: bool=False,
#             include_face: bool=True,
#             json_pose_callback: Callable[[str], None] = None,
#             res: int = 512,
#             **kwargs  # Ignore rest of kwargs
#     ) -> Tuple[np.ndarray, bool]:
#         """Run the openpose model. Returns a tuple of
#         - result image
#         - is_image flag

#         The JSON format pose string is passed to `json_pose_callback`.
#         """
#         print("调用OpenposeModel!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-------------------------------------------------------")
#         if json_pose_callback is None:
#             json_pose_callback = lambda x: None

#         img, remove_pad = resize_image_with_pad(img, res)

#         # if self.model_openpose is None:
#         #     from openpose import OpenposeDetector
#         #     self.model_openpose = OpenposeDetector()
#         ########################
#         model_openpose = self.model_openpose(
#             img,
#             include_body=include_body,
#             include_hand=include_hand,
#             include_face=include_face,
#             json_pose_callback=json_pose_callback
#         )
#         return self.openpose_pic, True


# g_openpose_model = OpenposeModel_original()
# def call_original(image_path):
#     # 打开图片
#     # image_path = '/root/openpose_test/1.png'
#     image = Image.open(image_path)
#     image_np = np.array(image)
#     # print(image_np)

#     start_time = time.time()

#     # 运行获得openpose图与计算结果
#     g_openpose_model.run_model(image_np)
    
#     # print(g_openpose_model.get_enable_Adtailer_variable)
#     end_time = time.time()
#     # 计算执行时间
#     execution_time = end_time - start_time
#     print('获得openpose图与计算结果花费时间：',execution_time)
    


# image_path = '../openpose_test/1.png'
# call_original(image_path)
