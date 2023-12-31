# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th Edited by ControlNet (Improved JSON serialization/deserialization, and lots of bug fixs)
# This preprocessor is licensed by CMU for non-commercial use only.


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import numpy as np
from . import util
from .body import Body, BodyResult, Keypoint
from .hand import Hand
from .face import Face
# from modules import devices
# from annotator_path import models_path

from typing import NamedTuple, Tuple, List, Callable, Union, Optional

body_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
hand_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth"
face_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth"

HandResult = List[Keypoint]
FaceResult = List[Keypoint]

class PoseResult(NamedTuple):
    body: BodyResult
    left_hand: Union[HandResult, None]
    right_hand: Union[HandResult, None]
    face: Union[FaceResult, None]

def draw_poses(poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        if draw_body:
            canvas = util.draw_bodypose(canvas, pose.body.keypoints)

        if draw_hand:
            canvas = util.draw_handpose(canvas, pose.left_hand)
            canvas = util.draw_handpose(canvas, pose.right_hand)

        if draw_face:
            canvas = util.draw_facepose(canvas, pose.face)

    return canvas


def decode_json_as_poses(json_string: str, normalize_coords: bool = False) -> Tuple[List[PoseResult], int, int]:
    """ Decode the json_string complying with the openpose JSON output format
    to poses that controlnet recognizes.
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md

    Args:
        json_string: The json string to decode.
        normalize_coords: Whether to normalize coordinates of each keypoint by canvas height/width.
                          `draw_pose` only accepts normalized keypoints. Set this param to True if
                          the input coords are not normalized.
    
    Returns:
        poses
        canvas_height
        canvas_width                      
    """
    pose_json = json.loads(json_string)
    height = pose_json['canvas_height']
    width = pose_json['canvas_width']

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    def normalize_keypoint(keypoint: Keypoint) -> Keypoint:
        return Keypoint(
            keypoint.x / width,
            keypoint.y / height
        )

    def decompress_keypoints(numbers: Optional[List[float]]) -> Optional[List[Optional[Keypoint]]]:
        if not numbers:
            return None
        
        assert len(numbers) % 3 == 0

        def create_keypoint(x, y, c):
            if c < 1.0:
                return None
            keypoint = Keypoint(x, y)
            if normalize_coords:
                keypoint = normalize_keypoint(keypoint)
            return keypoint

        return [
            create_keypoint(x, y, c)
            for x, y, c in chunks(numbers, n=3)
        ]
    
    return (
        [
            PoseResult(
                body=BodyResult(keypoints=decompress_keypoints(pose.get('pose_keypoints_2d'))),
                left_hand=decompress_keypoints(pose.get('hand_left_keypoints_2d')),
                right_hand=decompress_keypoints(pose.get('hand_right_keypoints_2d')),
                face=decompress_keypoints(pose.get('face_keypoints_2d'))
            )
            for pose in pose_json['people']
        ],
        height,
        width,
    )


def encode_poses_as_json(poses: List[PoseResult], canvas_height: int, canvas_width: int) -> str:
    """ Encode the pose as a JSON string following openpose JSON output format:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
    """
    def compress_keypoints(keypoints: Union[List[Keypoint], None]) -> Union[List[float], None]:
        if not keypoints:
            return None
        
        return [
            value
            for keypoint in keypoints
            for value in (
                [float(keypoint.x), float(keypoint.y), 1.0]
                if keypoint is not None
                else [0.0, 0.0, 0.0]
            )
        ]

    return json.dumps({
        'people': [
            {
                'pose_keypoints_2d': compress_keypoints(pose.body.keypoints),
                "face_keypoints_2d": compress_keypoints(pose.face),
                "hand_left_keypoints_2d": compress_keypoints(pose.left_hand),
                "hand_right_keypoints_2d":compress_keypoints(pose.right_hand),
            }
            for pose in poses
        ],
        'canvas_height': canvas_height,
        'canvas_width': canvas_width,
    }, indent=4)
    
    
class OpenposeDetector:
    """
    A class for detecting human poses in images using the Openpose model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """
    # model_dir = os.path.join(models_path, "openpose")
    model_dir = '../openpose_test/openpose_preprocessor'
    

    def __init__(self):
        # self.device = devices.get_device_for("controlnet")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.body_estimation = None
        self.hand_estimation = None
        self.face_estimation = None
        # [人数、min(x),max(x),min(y),max(y),人脸关键点,人脸面积]
        self.results={'people_num':0,'min_x':1,'max_x':0,'min_y':1,'max_y':0,'face_points':[],'area':[]}
        # self.result=[0,1,0,1,0,1,0]


    def load_model(self):
        """
        Load the Openpose body, hand, and face models.
        """
        body_modelpath = os.path.join(self.model_dir, "body_pose_model.pth")
        hand_modelpath = os.path.join(self.model_dir, "hand_pose_model.pth")
        face_modelpath = os.path.join(self.model_dir, "facenet.pth")

        if not os.path.exists(body_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(body_model_path, model_dir=self.model_dir)

        if not os.path.exists(hand_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(hand_model_path, model_dir=self.model_dir)

        if not os.path.exists(face_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(face_model_path, model_dir=self.model_dir)

        self.body_estimation = Body(body_modelpath)
        self.hand_estimation = Hand(hand_modelpath)
        self.face_estimation = Face(face_modelpath)

    def unload_model(self):
        """
        Unload the Openpose models by moving them to the CPU.
        """
        if self.body_estimation is not None:
            self.body_estimation.model.to("cpu")
            self.hand_estimation.model.to("cpu")
            self.face_estimation.model.to("cpu")

    def detect_hands(self, body: BodyResult, oriImg) -> Tuple[Union[HandResult, None], Union[HandResult, None]]:
        left_hand = None
        right_hand = None
        H, W, _ = oriImg.shape
        for x, y, w, is_left in util.handDetect(body, oriImg):
            peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :]).astype(np.float32)
            if peaks.ndim == 2 and peaks.shape[1] == 2:
                peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                
                hand_result = [
                    Keypoint(x=peak[0], y=peak[1])
                    for peak in peaks
                ]

                if is_left:
                    left_hand = hand_result
                else:
                    right_hand = hand_result

        return left_hand, right_hand

    def detect_face(self, body: BodyResult, oriImg) -> Union[FaceResult, None]:
        face = util.faceDetect(body, oriImg)
        if face is None:
            return None
        face_points=0
        x, y, w = face
        H, W, _ = oriImg.shape
        heatmaps = self.face_estimation(oriImg[y:y+w, x:x+w, :])
        peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(np.float32)
        if peaks.ndim == 2 and peaks.shape[1] == 2:
            peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
            peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            for i in [
                Keypoint(x=peak[0], y=peak[1])
                for peak in peaks
            ]:
                # # [人数、min(x),max(x),min(y),max(y),is_Nonem人脸关键点]
                self.results['min_x']=min(self.results['min_x'],i.x)
                self.results['max_x']=max(self.results['max_x'],i.x)
                self.results['min_y']=min(self.results['min_y'],i.y)
                self.results['max_y']=max(self.results['max_y'],i.y)
                face_points += 1
                # self.results['face_points']=len([Keypoint(x=peak[0], y=peak[1]) for peak in peaks])
                # self.result=[0,1,0,1,0,1,0]
                # self.result[1]=min(self.result[1],i.x)
                # self.result[2]=max(self.result[2],i.x)
                # self.result[3]=min(self.result[3],i.y)
                # self.result[4]=max(self.result[4],i.y)
            #     self.result[6]=len([
            #     Keypoint(x=peak[0], y=peak[1])
            #     for peak in peaks])
            area = self.calculate_face_area()
            self.results['area'].append(area)
            # 坐标清零
            self.results['min_x']=1
            self.results['max_x']=0
            self.results['min_y']=1
            self.results['max_y']=0
            self.results['face_points'].append(face_points)

            print('face_points:',len([
                Keypoint(x=peak[0], y=peak[1])
                for peak in peaks]))
            return [
                Keypoint(x=peak[0], y=peak[1])
                for peak in peaks
            ]
        
        return None

    def detect_poses(self, oriImg, include_hand=False, include_face=False) -> List[PoseResult]:
        """
        Detect poses in the given image.
            Args:
                oriImg (numpy.ndarray): The input image for pose detection.
                include_hand (bool, optional): Whether to include hand detection. Defaults to False.
                include_face (bool, optional): Whether to include face detection. Defaults to False.

        Returns:
            List[PoseResult]: A list of PoseResult objects containing the detected poses.
        """
        if self.body_estimation is None:
            self.load_model()
            
        self.body_estimation.model.to(self.device)
        self.hand_estimation.model.to(self.device)
        self.face_estimation.model.to(self.device)

        self.body_estimation.cn_device = self.device
        self.hand_estimation.cn_device = self.device
        self.face_estimation.cn_device = self.device

        oriImg = oriImg[:, :, ::-1].copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            bodies = self.body_estimation.format_body_result(candidate, subset)

            results = []
            for body in bodies:
                left_hand, right_hand, face = (None,) * 3
                if include_hand:
                    left_hand, right_hand = self.detect_hands(body, oriImg)
                if include_face:
                    face = self.detect_face(body, oriImg)
                    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    # print('face:',face)
                    # print('\n\n')
                
                results.append(PoseResult(BodyResult(
                    keypoints=[
                        Keypoint(
                            x=keypoint.x / float(W),
                            y=keypoint.y / float(H)
                        ) if keypoint is not None else None
                        for keypoint in body.keypoints
                    ], 
                    total_score=body.total_score,
                    total_parts=body.total_parts
                ), left_hand, right_hand, face))
                # ###############################
            self.results['people_num']=len(results)
            return results
        
    def __call__(
            self, oriImg, include_body=True, include_hand=False, include_face=False,
            json_pose_callback: Callable[[str], None] = None,
        ):
        """
        Detect and draw poses in the given image.

        Args:
            oriImg (numpy.ndarray): The input image for pose detection and drawing.
            include_body (bool, optional): Whether to include body keypoints. Defaults to True.
            include_hand (bool, optional): Whether to include hand keypoints. Defaults to False.
            include_face (bool, optional): Whether to include face keypoints. Defaults to False.
            json_pose_callback (Callable, optional): A callback that accepts the pose JSON string.

        Returns:
            numpy.ndarray: The image with detected and drawn poses.
        """
        H, W, _ = oriImg.shape
        poses = self.detect_poses(oriImg, include_hand, include_face)
        # print('----------------------------------------------------------')
        # print('poses:',poses)
        if json_pose_callback:
            json_pose_callback(encode_poses_as_json(poses, H, W))

        # if self.result[0]==0:
        #     self.enable_ADtailer=False
        # # 脸部面积占整张图片的面积大于1/10，脸部相对完整,则不需要调用ADtailer
        # elif self.result[0]==1 and self.calculate_face_area()>0.1 and self.result[6]==70:
        #     self.enable_ADtailer=False
        # print(self.enable_ADtailer)
        return draw_poses(poses, H, W, draw_body=include_body, draw_hand=include_hand, draw_face=include_face) 
    
    
    def calculate_face_area(self):
        
        # print(abs((self.result[2]-self.result[1])*(self.result[4]-self.result[3])))
        return abs((self.results['max_x']-self.results['min_x'])*(self.results['max_y']-self.results['min_y']))

    def initialize_attribute(self):
        # self.body_estimation = None
        # self.hand_estimation = None
        # self.face_estimation = None
        # [人数、min(x),max(x),min(y),max(y),人脸关键点,人脸面积]
        # self.results={'people_num':0,'min_x':1,'max_x':0,'min_y':1,'max_y':0,'face_points':[],'area':[]}
        self.results['face_points']=[]
        self.results['area']=[]



class OpenposeDetector_original:
    """
    A class for detecting human poses in images using the Openpose model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """
    # model_dir = os.path.join(models_path, "openpose")
    model_dir = '../openpose_test/openpose_preprocessor'
    

    def __init__(self):
        # self.device = devices.get_device_for("controlnet")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.body_estimation = None
        self.hand_estimation = None
        self.face_estimation = None
        # # [人数、min(x),max(x),min(y),max(y),人脸关键点,人脸面积]
        # self.results={'people_num':0,'min_x':1,'max_x':0,'min_y':1,'max_y':0,'face_points':[],'area':[]}
        # # self.result=[0,1,0,1,0,1,0]
        # # self.enable_ADtailer=True

    def load_model(self):
        """
        Load the Openpose body, hand, and face models.
        """
        body_modelpath = os.path.join(self.model_dir, "body_pose_model.pth")
        hand_modelpath = os.path.join(self.model_dir, "hand_pose_model.pth")
        face_modelpath = os.path.join(self.model_dir, "facenet.pth")

        if not os.path.exists(body_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(body_model_path, model_dir=self.model_dir)

        if not os.path.exists(hand_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(hand_model_path, model_dir=self.model_dir)

        if not os.path.exists(face_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(face_model_path, model_dir=self.model_dir)

        self.body_estimation = Body(body_modelpath)
        self.hand_estimation = Hand(hand_modelpath)
        self.face_estimation = Face(face_modelpath)

    def unload_model(self):
        """
        Unload the Openpose models by moving them to the CPU.
        """
        if self.body_estimation is not None:
            self.body_estimation.model.to("cpu")
            self.hand_estimation.model.to("cpu")
            self.face_estimation.model.to("cpu")

    def detect_hands(self, body: BodyResult, oriImg) -> Tuple[Union[HandResult, None], Union[HandResult, None]]:
        left_hand = None
        right_hand = None
        H, W, _ = oriImg.shape
        for x, y, w, is_left in util.handDetect(body, oriImg):
            peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :]).astype(np.float32)
            if peaks.ndim == 2 and peaks.shape[1] == 2:
                peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                
                hand_result = [
                    Keypoint(x=peak[0], y=peak[1])
                    for peak in peaks
                ]

                if is_left:
                    left_hand = hand_result
                else:
                    right_hand = hand_result

        return left_hand, right_hand

    def detect_face(self, body: BodyResult, oriImg) -> Union[FaceResult, None]:
        face = util.faceDetect(body, oriImg)
        if face is None:
            return None
        face_points=0
        x, y, w = face
        H, W, _ = oriImg.shape
        heatmaps = self.face_estimation(oriImg[y:y+w, x:x+w, :])
        peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(np.float32)
        if peaks.ndim == 2 and peaks.shape[1] == 2:
            peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
            peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            # for i in [
            #     Keypoint(x=peak[0], y=peak[1])
            #     for peak in peaks
            # ]:
            #     # # [人数、min(x),max(x),min(y),max(y),is_Nonem人脸关键点]
            #     self.results['min_x']=min(self.results['min_x'],i.x)
            #     self.results['max_x']=max(self.results['max_x'],i.x)
            #     self.results['min_y']=min(self.results['min_y'],i.y)
            #     self.results['max_y']=max(self.results['max_y'],i.y)
            #     face_points += 1
            #     # self.results['face_points']=len([Keypoint(x=peak[0], y=peak[1]) for peak in peaks])
            #     # self.result=[0,1,0,1,0,1,0]
            #     # self.result[1]=min(self.result[1],i.x)
            #     # self.result[2]=max(self.result[2],i.x)
            #     # self.result[3]=min(self.result[3],i.y)
            #     # self.result[4]=max(self.result[4],i.y)
            # #     self.result[6]=len([
            # #     Keypoint(x=peak[0], y=peak[1])
            # #     for peak in peaks])
            # area = self.calculate_face_area()
            # self.results['area'].append(area)
            # # 坐标清零
            # self.results['min_x']=1
            # self.results['max_x']=0
            # self.results['min_y']=1
            # self.results['max_y']=0
            # self.results['face_points'].append(face_points)

            # print('face_points:',len([
            #     Keypoint(x=peak[0], y=peak[1])
            #     for peak in peaks]))
            return [
                Keypoint(x=peak[0], y=peak[1])
                for peak in peaks
            ]
        
        return None

    def detect_poses(self, oriImg, include_hand=False, include_face=False) -> List[PoseResult]:
        """
        Detect poses in the given image.
            Args:
                oriImg (numpy.ndarray): The input image for pose detection.
                include_hand (bool, optional): Whether to include hand detection. Defaults to False.
                include_face (bool, optional): Whether to include face detection. Defaults to False.

        Returns:
            List[PoseResult]: A list of PoseResult objects containing the detected poses.
        """
        if self.body_estimation is None:
            self.load_model()
            
        self.body_estimation.model.to(self.device)
        self.hand_estimation.model.to(self.device)
        self.face_estimation.model.to(self.device)

        self.body_estimation.cn_device = self.device
        self.hand_estimation.cn_device = self.device
        self.face_estimation.cn_device = self.device

        oriImg = oriImg[:, :, ::-1].copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            bodies = self.body_estimation.format_body_result(candidate, subset)

            results = []
            for body in bodies:
                left_hand, right_hand, face = (None,) * 3
                if include_hand:
                    left_hand, right_hand = self.detect_hands(body, oriImg)
                if include_face:
                    face = self.detect_face(body, oriImg)
                    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    # print('face:',face)
                    # print('\n\n')
                
                results.append(PoseResult(BodyResult(
                    keypoints=[
                        Keypoint(
                            x=keypoint.x / float(W),
                            y=keypoint.y / float(H)
                        ) if keypoint is not None else None
                        for keypoint in body.keypoints
                    ], 
                    total_score=body.total_score,
                    total_parts=body.total_parts
                ), left_hand, right_hand, face))
                # ###############################
            # self.results['people_num']=len(results)
            return results
        
    def __call__(
            self, oriImg, include_body=True, include_hand=False, include_face=False,
            json_pose_callback: Callable[[str], None] = None,
        ):
        """
        Detect and draw poses in the given image.

        Args:
            oriImg (numpy.ndarray): The input image for pose detection and drawing.
            include_body (bool, optional): Whether to include body keypoints. Defaults to True.
            include_hand (bool, optional): Whether to include hand keypoints. Defaults to False.
            include_face (bool, optional): Whether to include face keypoints. Defaults to False.
            json_pose_callback (Callable, optional): A callback that accepts the pose JSON string.

        Returns:
            numpy.ndarray: The image with detected and drawn poses.
        """
        H, W, _ = oriImg.shape
        poses = self.detect_poses(oriImg, include_hand, include_face)
        # print('----------------------------------------------------------')
        # print('poses:',poses)
        if json_pose_callback:
            json_pose_callback(encode_poses_as_json(poses, H, W))

        # if self.result[0]==0:
        #     self.enable_ADtailer=False
        # # 脸部面积占整张图片的面积大于1/10，脸部相对完整,则不需要调用ADtailer
        # elif self.result[0]==1 and self.calculate_face_area()>0.1 and self.result[6]==70:
        #     self.enable_ADtailer=False
        # print(self.enable_ADtailer)
        return draw_poses(poses, H, W, draw_body=include_body, draw_hand=include_hand, draw_face=include_face) 
    
    
    # def calculate_face_area(self):
        
    #     # print(abs((self.result[2]-self.result[1])*(self.result[4]-self.result[3])))
    #     return abs((self.results['max_x']-self.results['min_x'])*(self.results['max_y']-self.results['min_y']))

    # def initialize_attribute(self):
    #     # self.body_estimation = None
    #     # self.hand_estimation = None
    #     # self.face_estimation = None
    #     # [人数、min(x),max(x),min(y),max(y),人脸关键点,人脸面积]
    #     # self.results={'people_num':0,'min_x':1,'max_x':0,'min_y':1,'max_y':0,'face_points':[],'area':[]}
    #     self.results['face_points']=[]
    #     self.results['area']=[]

