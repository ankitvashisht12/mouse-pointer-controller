from face_detection import Model_Face_Detection
from facial_landmark_detection import Model_Facial_Landmark_Detection
from gaze_estimation import Model_Gaze_Estimation
from head_pose_estimation import Model_Head_Pose_Estimation
from input_feeder import InputFeeder
from mouse_controller import MouseController
import cv2
from argparse import ArgumentParser
import numpy as np

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-m1", "--face_detection", required=True, type=str, help="Path to face detection")
    parser.add_argument("-m2", "--landmark_detection", required=True, type=str, help="Path to landmark detection model")
    parser.add_argument("-m3", "--head_pose_detection", required=True, type=str, help="Path to head pose detection model")
    # parser.add_argument("-m4", "--gaze_detection", required=True, type=str, help="Path to gaze detection model")
    parser.add_argument("-d", "--device", required=False, type=str, help="Specify device", default="CPU")
    parser.add_argument("-e", "--extention", required=False, type=str, help="Specify CPU extention", default=None)
    parser.add_argument("-i", "--input_file", required=True, type=str, help="Specify input file type", default="cam")
    parser.add_argument("-p", "--input_path", required=False, type=str, help="Specify input file path", default=None)
    return parser.parse_args()



def main():

    # get arguments
    args =  get_args()
   
    # Initialize models
    fd = Model_Face_Detection(args.face_detection, args.device, args.extention)
    ld = Model_Facial_Landmark_Detection(args.landmark_detection, args.device, args.extention)
    hp = Model_Head_Pose_Estimation(args.head_pose_detection, args.device, args.extention)
    #gd = Model_Gaze_Estimation(args.gaze_detection, args.device, args.extention)

    # load models
    fd.load_model()
    ld.load_model()
    hp.load_model()
    #gd.load_model()

    # Initialize input feed and load data
    input_feed = InputFeeder(args.input_file, args.input_path)
    input_feed.load_data()


    for frame in input_feed.next_batch():
        
        outs_fd = fd.predict(frame) 
        cropped_face = []
      
        if len(outs_fd) != 0:

            xmin = outs_fd[0][0]
            ymin = outs_fd[0][1]
            xmax = outs_fd[0][2]
            ymax = outs_fd[0][3]
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            color = (255, 0, 0)

            show_frame = cv2.rectangle(img=frame, pt1=start_point, pt2=end_point , color=color , thickness=2)

            delta_y = ymax - ymin
            delta_x = xmax - xmin
            cropped_face = frame[ymin:ymin+delta_y, xmin:xmin+delta_x]

            # Uncomment to see cropped face
            # cv2.imshow("Cropped Face", cropped_face)
            # cv2.waitKey(5)

           

            outs_ld = ld.predict(cropped_face.copy())
            if len(outs_ld) != 0:
            

                p1 = tuple(sum(x) for x in zip(outs_ld[0][0], start_point)) 
                p2 = tuple(sum(x) for x in zip(outs_ld[0][1], start_point))
                p3 = tuple(sum(x) for x in zip(outs_ld[0][2], start_point))
                p4 = tuple(sum(x) for x in zip(outs_ld[0][3], start_point))
                
                show_frame = cv2.circle(show_frame, p1, 2, (0, 0, 255), thickness=-5)
                show_frame = cv2.circle(show_frame, p2, 2, (0, 0, 255), thickness=-5)
                show_frame = cv2.circle(show_frame, p3, 2, (0, 0, 255), thickness=-5)
                show_frame = cv2.circle(show_frame, p4, 2, (0, 0, 255), thickness=-5)
            
            
            p, r, y = hp.predict(cropped_face.copy())  # pitch, roll, yaw
                

        else:
            show_frame = frame
        
        # Uncomment for video stream with bounding box on face
        # cv2.imshow("Capturing", show_frame)
        # key = cv2.waitKey(1)
        # if key == ord('q'):
        #     break

        # ## Control Mouse pointer 
        # mc = MouseController(???, ???)
        # mc.move()

    input_feed.close()


if __name__ == "__main__":
    main()