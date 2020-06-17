from face_detection import Model_Face_Detection
from facial_landmark_detection import Model_Facial_Landmark_Detection
from gaze_estimation import Model_Gaze_Estimation
from head_pose_estimation import Model_Head_Pose_Estimation
from input_feeder import InputFeeder
from mouse_controller import MouseController
import cv2
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-m1", "--face_detection", required=True, type=str, help="Path to face detection")
    parser.add_argument("-m2", "--landmark_detection", required=True, type=str, help="Path to landmark detection model")
    parser.add_argument("-m3", "--head_pose_detection", required=True, type=str, help="Path to head pose detection model")
    parser.add_argument("-m4", "--gaze_detection", required=True, type=str, help="Path to gaze detection model")
    parser.add_argument("-d", "--device", required=False, type=str, help="Specify device", default="CPU")
    parser.add_argument("-e", "--extention", required=False, type=str, help="Specify CPU extention", default=None)
    parser.add_argument("-i", "--input_file", required=True, type=str, help="Specify input file type", default="cam")
    parser.add_argument("-p", "--input_path", required=False, type=str, help="Specify input file path")
    return parser.parse_args()



def main():
    args =  get_args()
    input_feed = InputFeeder('cam')
    input_feed.load_data()
    for frame in input_feed.next_batch():
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        fd = Model_Face_Detection(args.face_detection, args.device, args.extention)
        fd.load_model()
        fd.preprocess_input(frame)
        outs_fd = fd.predict()

        ld = Model_Facial_Landmark_Detection(args.landmark_detection, args.device, args.extention)
        ld.load_model()
        ld.preprocess_input(???)  # TODO: Check input - outs_fd
        outs_ld = ld.predict()

        hp = Model_Head_Pose_Estimation()
        hp.load_model()
        hp.preprocess_input(???) # TODO: Check input - outs_fd
        outs_hp = hp.predict()

        gd = Model_Gaze_Estimation()
        gd.load_model()
        gd.preprocess_input(???)
        outs_gd = gd.predict()

        ## Control Mouse pointer 
        mc = MouseController(???, ???)
        mc.move()

    input_feed.close()


if __name__ == "__main__":
    main()