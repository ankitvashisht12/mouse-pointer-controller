from face_detection import Model_Face_Detection
from facial_landmark_detection import Model_Facial_Landmark_Detection
from gaze_estimation import Model_Gaze_Estimation
from head_pose_estimation import Model_Head_Pose_Estimation
from input_feeder import InputFeeder
from mouse_controller import MouseController


def main():
    input_feed = InputFeeder('cam')
    input_feed.load_data()
    for batch in input_feed.next_batch():
        pass
    input_feed.close()


if __name__ == "__main__":
    main()