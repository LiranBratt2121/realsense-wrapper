import pyrealsense2 as rs
from ultralytics import YOLO
import supervision as sv
from enum import Enum
import numpy as np
import cv2

class RealsenseCamera:
    """
    ## RealsenseCamera class for interfacing with a RealSense camera and processing IMU data.
    ----
    Attributes:
    ----
        - ALPHA (float): Weight for combining gyro and accelerometer angles.
        - RAD_TO_DEG (float): Conversion factor from radians to degrees.
        - PI (float): Value of pi.

    Methods:
    ----
        - initialize_camera: Initializes the RealSense camera pipeline and configuration.
        - update: Waits for the next set of frames from the camera and updates internal frames.
        - process: Processes the depth and color frames, and calculates yaw, pitch, and roll angles.
        - get_yaw_from_target: Calculates the yaw angle from the camera to the target
        - get_distance_from_target: Calculates the linear distance from the camera to the target
    """

    ALPHA = 0.98
    RAD_TO_DEG = 57.2958
    PI = np.pi
    
    class ModelType(Enum):
        BEST = 0
        LAST = 1

    def __init__(self) -> None:
        """
        ## Initializes a RealsenseCamera object.
        ----
        Attributes:
        ----
            - pipe (rs.pipeline()): RealSense pipeline for capturing frames.
            - cfg (rs.config()): RealSense configuration for configuring stream settings.
            - focal_length (float): Focal length of the camera
            - image_size(tuple): The size of the image
            - frame(rs.composite_frame()): Current frame from the RealSense camera.
            - depth_frame(rs.depth_frame()): Depth frame captured by the camera.
            - color_frame(rs.video_frame()): Color frame captured by the camera.
            - depth_image(np.NDArray): Numpy array representing the depth image.
            - color_image(np.NDArray): Numpy array representing the color image.
            - depth_color_map(cv2.UMat): Color-mapped depth image.
            - yaw(float): Realsens's yaw angle.
            - pitch(float): Realsens's pitch angle.
            - roll(float): Realsens's roll angle.

        Private Attrbutes:
        ----
            - __acceleration_frame(rs.frame()): Frame containing accelerometer data.
            - __gyro_frame(rs.frame()): Frame containing gyrometer data.
            - __acceleration(tuple[float]): Acceleration data.
            - __gyro(tupple[float]): Gyro data.
            - __accel_angle_z(float): Accelerometer angle around the Z-axis.
            - __accel_angle_y(float): Accelerometer angle around the Y-axis.
            - __accel_angle_x(float): Accelerometer angle around the X-axis.
            - __total_gyro_angleZ(float): Total gyrometer angle around the Z-axis.
            - __total_gyro_angleY(float): Total gyrometer angle around the Y-axis.
            - __total_gyro_angleX(float): Total gyrometer angle around the X-axis.
            - __first (bool): Flag indicating whether it is the first frame.
        """
        self.pipe, self.cfg = self.initialize_camera()

        self.focal_length, self.image_size = self.__get_intrinsics()
        
        self.frame = rs.composite_frame
        self.depth_frame = rs.depth_frame
        self.color_frame = rs.video_frame

        self.depth_image = np.ndarray
        self.color_image = np.ndarray
        self.depth_color_map = cv2.UMat
        
        self.__acceleration_frame = rs.frame
        self.__gyro_frame = rs.frame
        
        self.__acceleration = float()
        self.__gyro = float()
        
        self.__accel_angle_z = float()
        self.__accel_angle_y = float()
        self.__accel_angle_x = float()
        
        self.__total_gyro_angleZ = 0
        self.__total_gyro_angleY = 0
        self.__total_gyro_angleX = 0

        self.__first = True
        
        self.yaw = float() 
        self.pitch = float() 
        self.roll = float()
        
        self.__model = YOLO()
        
                
    def initialize_camera(self) -> tuple[rs.pipeline, rs.config]:
        """
        ## Initializes the RealSense camera pipeline and configuration.
        ----

        Returns:
        ----
            - tuple: RealSense pipeline and configuration.
        """
        pipe = rs.pipeline()
        cfg = rs.config()
        
        cfg.enable_stream(rs.stream.accel)
        cfg.enable_stream(rs.stream.gyro)
        
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        pipe.start(cfg)
        
        return pipe, cfg

    
    def update(self) -> None:
        """
        Waits for the next set of frames from the camera and updates internal frames.
        """
        self.frame = self.pipe.wait_for_frames()

        self.depth_frame = self.frame.get_depth_frame()
        self.color_frame = self.frame.get_color_frame()

        self.__acceleration_frame = self.frame.first_or_default(rs.stream.accel, rs.format.motion_xyz32f)
        self.__gyro_frame = self.frame.first_or_default(rs.stream.gyro, rs.format.motion_xyz32f)


    def process(self) -> None:
        """
        ## Processes the depth and color frames and calculates yaw, pitch, and roll angles.
        ----
        """
        self.update()
        
        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.color_image = np.asanyarray(self.color_frame.get_data())
        self.depth_color_map = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.1), cv2.COLORMAP_JET)
        
        self.__acceleration = self.__acceleration_frame.as_motion_frame().get_motion_data()
        self.__gyro = self.__gyro_frame.as_motion_frame().get_motion_data()
        
        self.__process_angles()
        
    
    def __process_angles(self):
        """
        ## Calculates yaw, pitch, and roll angles based on gyrometer and accelerometer data.
        ----
        """
        if self.__first:
            self.__first = False
            self.last_ts_gyro = self.__gyro_frame.get_timestamp()

            # accelerometer calculation
            self.__accel_angle_z = np.degrees(np.arctan2(self.__acceleration.y, self.__acceleration.z))
            self.__accel_angle_x = np.degrees(np.arctan2(self.__acceleration.x, np.sqrt(self.__acceleration.y ** 2 + self.__acceleration.z ** 2)))
            self.__accel_angle_y = np.degrees(self.PI)

            return

        # gyro calculations
        dt_gyro = (self.__gyro_frame.get_timestamp() - self.last_ts_gyro) / 1000
        self.last_ts_gyro = self.__gyro_frame.get_timestamp()

        gyro_angle_x = self.__gyro.x * dt_gyro
        gyro_angle_y = self.__gyro.y * dt_gyro
        gyro_angle_z = self.__gyro.z * dt_gyro

        dangleX = gyro_angle_x * self.RAD_TO_DEG
        dangleY = gyro_angle_y * self.RAD_TO_DEG
        dangleZ = gyro_angle_z * self.RAD_TO_DEG

        self.__total_gyro_angleX = self.__accel_angle_x + dangleX
        self.__total_gyro_angleY = self.__accel_angle_y + dangleY + self.__total_gyro_angleY
        self.__total_gyro_angleZ = self.__accel_angle_z + dangleZ

        # accelerometer calculation
        self.__accel_angle_z = np.degrees(np.arctan2(self.__acceleration.y, self.__acceleration.z))
        self.__accel_angle_x = np.degrees(np.arctan2(self.__acceleration.x, np.sqrt(self.__acceleration.y ** 2 + self.__acceleration.z ** 2)))
        self.__accel_angle_y = 0

        # combining gyrometer and accelerometer angles
        combinedangleX = self.__total_gyro_angleX * self.ALPHA + self.__accel_angle_x * (1 - self.ALPHA)
        combinedangleZ = self.__total_gyro_angleZ * self.ALPHA + self.__accel_angle_z * (1 - self.ALPHA)
        combinedangleY = self.__total_gyro_angleY

        # Update yaw, pitch, and roll angles
        self.yaw = combinedangleY - 180
        self.pitch = combinedangleZ + 90
        self.roll = combinedangleX
    
    
    def get_yaw_from_target(self, target_pixel_coordinates: tuple[int, int]) -> float:
        """
        ## Calculate the yaw angle in degrees from the RealSense camera to a target pixel.
        ----
        Parameters:
        ----
            - target_pixel_coordinates (tuple[int, int]): The pixel coordinates (x, y) of the target in the image.

        Returns:
        ----
            - float: The yaw angle in degrees.

        Note:
        ----
        - The function uses the depth information from the RealSense depth frame to calculate the horizontal angle
        between the target pixel and the center of the image.
        - The calculation is based on the arctangent of the horizontal displacement divided by the focal length.
        - The result is then converted to degrees and returned as the yaw angle.
        """
        center_pixel = self.image_size[1] // 2
        
        # Calculate horizontal angle
        horizontal_angle: float = np.arctan2(
            target_pixel_coordinates[0] - center_pixel, self.focal_length)

        yaw_angle: float = np.degrees(horizontal_angle)

        return yaw_angle
    
    
    def get_distance_from_target(self, target_pixel_coordinates: tuple[int, int]) -> float:
        """
        ## Calculate the distance in meters from the RealSense camera to a target pixel.
        ----
        Parameters:
        ----
            - target_pixel_coordinates (tuple[int, int]): The pixel coordinates (x, y) of the target in the image.

        Returns:
        ----
            - float: The distance in meters.

        Note:
        ----
        - The function uses the depth information from the RealSense depth frame to calculate the linear distance
        between the camera and the target pixel. You may need to use so trig to find the true distance to the object using the camera pitch
        """
        distance = self.depth_frame.get_distance(target_pixel_coordinates[0], target_pixel_coordinates[1])

        return distance
    
    
    def __get_intrinsics(self) -> tuple[float, tuple[float, float]]:
        """
        ## Returns the Realsense's focal length and image size
        ----
        
        Returns:
        ----
        - Tuple[str, float] = (focal_length, image_size)
        """
        profile: rs.pipeline_profile = self.pipe.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        intrinsics = depth_profile.get_intrinsics()

        fx: float = intrinsics.fx
        fy: float = intrinsics.fy

        focal_length = (fx + fy) / 2
        
        height: float = intrinsics.height
        width: float = intrinsics.width
        
        image_size = (height, width)
        
        return focal_length, image_size
    
    
    def initialize_model(self, model_type: ModelType) -> None:
        self.__model = YOLO('Best' if model_type.value == 0 else 'LAST')
        
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.__model(frame)[0]
        
        detections = sv.Detections.from_yolov8(results)

        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

        labels = [f"{self.__model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        return frame
    

def main():
    camera = RealsenseCamera()
    
    while True:
        camera.process()
        
        cv2.putText(camera.color_image, f"Yaw: {camera.yaw:.2f} Pitch: {camera.pitch:.2f} Roll: {camera.roll:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        
        cv2.imshow("Depth", camera.depth_color_map)
        cv2.imshow("Color", camera.color_image)
        
        if cv2.waitKey(1) & ord('q') == 0xFF:
            cv2.destroyAllWindows()
            camera.pipe.stop()
            break

if __name__ == '__main__':
    main()
