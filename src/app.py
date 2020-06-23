# Imports
import logging
import time
from face_detection import Model_Face_Detection
from facial_landmark_detection import Model_Facial_Landmark_Detection
from gaze_estimation import Model_Gaze_Estimation
from head_pose_estimation import Model_Head_Pose_Estimation
from input_feeder import InputFeeder
from mouse_controller import MouseController
import cv2
import os
from argparse import ArgumentParser
import numpy as np

def get_args():
    parser = ArgumentParser(description='Mouse Pointer Controller using eye gaze')
    parser.add_argument("-m1", "--face_detection", required=True, type=str, help="Path to face detection")
    parser.add_argument("-m2", "--landmark_detection", required=True, type=str, help="Path to landmark detection model")
    parser.add_argument("-m3", "--head_pose_detection", required=True, type=str, help="Path to head pose detection model")
    parser.add_argument("-m4", "--gaze_detection", required=True, type=str, help="Path to gaze detection model")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-e", "--extention", required=False, type=str, help="Specify CPU extention", default=None)
    parser.add_argument("-i", "--input_file", required=True, type=str, help="Specify input file type", default="cam")
    parser.add_argument("-p", "--input_path", required=False, type=str, help="Specify input file path", default=None)
    parser.add_argument("-v", "--visualize", required=False, type=str, nargs='+', help="Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame," 
                             "fd for Face Detection, ld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation.", default = [])
    parser.add_argument("-prob", "--prob", required=False, type=float, help="threshod probability")
    return parser.parse_args()

def crop_eyes(frame, coords):
   
    left_eye = frame[coords[0][1] : coords[1][1] , coords[0][0] : coords[1][0]]
    right_eye = frame[coords[2][1] : coords[3][1] , coords[2][0] : coords[3][0]]
   
    return left_eye, right_eye

def show_visualization(frame, visualization_list, start_point, end_point, eye_bb, eye_coords, hp, ge):
    # draw bounding box around the face
    if 'fd' in visualization_list:
        frame = cv2.rectangle(img=frame, pt1=start_point, pt2=end_point , color=(0, 255, 0) , thickness=2)
    
    # draw facial landmarks on the face 
    if 'ld' in visualization_list:
        # draw circle on the detected coordinates of left and right eye
        frame = cv2.circle(frame, eye_coords[0], 2, (0, 0, 255), thickness=-5)
        frame = cv2.circle(frame, eye_coords[1], 2, (0, 0, 255), thickness=-5)
        frame = cv2.circle(frame, eye_coords[2], 2, (0, 0, 255), thickness=-5)
        frame = cv2.circle(frame, eye_coords[3], 2, (0, 0, 255), thickness=-5)

        # draw bounding box around the eyes
        frame = cv2.rectangle(img = frame, pt1=eye_bb[0], pt2=eye_bb[1], color=(255, 0, 0), thickness=2)
        frame = cv2.rectangle(img = frame, pt1=eye_bb[2], pt2=eye_bb[3], color=(255, 0, 0), thickness=2)

        if 'hp' in visualization_list:
            cv2.putText(frame, "Pose Angles: pitch:{:.2f} , roll:{:.2f} , yaw:{:.2f}".format(hp[0],hp[1],hp[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 0, 255), 1)
        
        if 'ge' in visualization_list:
            # x, y, w = int(ge[0]*12), int(ge[1]*12), 160
            # le =cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
            # cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
            # re = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
            # cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
            pass
            


  
    cv2.imshow("Visualizations", cv2.resize(frame, (700, 500)))


def crop_face(start_point, end_point, frame):
    delta_y = end_point[1] - start_point[1]
    delta_x = end_point[0] - start_point[0]
    return frame[start_point[1]:start_point[1]+delta_y, start_point[0]:start_point[0]+delta_x]


def main():

    # get arguments
    args =  get_args()
    visualization_list = args.visualize
    prob = args.prob
    if prob == None:
        prob = 0.5
    input_type = args.input_file
    input_path = args.input_path

    #logging config
    
    logging.basicConfig(filename="app.log", level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

    # Initialize models
    try:
        fd = Model_Face_Detection(args.face_detection, args.device, args.extention)
        ld = Model_Facial_Landmark_Detection(args.landmark_detection, args.device, args.extention)
        hp = Model_Head_Pose_Estimation(args.head_pose_detection, args.device, args.extention)
        gd = Model_Gaze_Estimation(args.gaze_detection, args.device, args.extention)
    except:
        logging.error("Error in initializing models")
        exit(1)

    # load models
    try:
        start_loading_time_fd = time.time()
        fd.load_model()
        fd_time_diff = time.time() - start_loading_time_fd
        start_loading_time_ld = time.time()
        ld.load_model()
        ld_time_diff = time.time() - start_loading_time_ld
        start_loading_time_hp = time.time()
        hp.load_model()
        hp_time_diff = time.time() - start_loading_time_hp
        start_loading_time_gd = time.time()
        gd.load_model()
        gd_time_diff = time.time() - start_loading_time_gd
    except:
        logging.error("Error in loading the models")
        exit(1)

    logging.debug("Loading times are facial detection : {} , landmark detection : {} , head pose detection : {} , gaze estimation : {} ".format(fd_time_diff, ld_time_diff, hp_time_diff, gd_time_diff))

    if input_type.lower() != "cam":
        if not os.path.isfile(input_path):
            logging.error("Unable to find specified video file")
            exit(1)
    else:
        input_path=None

    # Initialize input feed and load data
    input_feed = InputFeeder(input_type, input_path)
    input_feed.load_data()


    avg_inf_time = {"fd":[], "ld":[], "hp":[], "gd":[]}

    for ret, frame in input_feed.next_batch():
        
        
        if not ret:
            break

        show_frame =frame
        outs_fd, fd_inf_time = fd.predict(frame.copy(), prob) 
      
        if len(outs_fd) == 0:
            continue
        
        start_point = outs_fd[0]
        end_point = outs_fd[1]

        cropped_face = crop_face(start_point, end_point, frame)

           
        # predict facial landmark on cropped image 
        outs_ld, ld_inf_time = ld.predict(cropped_face.copy())
        if len(outs_ld) == 0:
            continue
                
        # extract coordinates for left and right eye 
        p1 = tuple(sum(x) for x in zip(outs_ld[0][0], start_point)) 
        p2 = tuple(sum(x) for x in zip(outs_ld[0][1], start_point))
        p3 = tuple(sum(x) for x in zip(outs_ld[0][2], start_point))
        p4 = tuple(sum(x) for x in zip(outs_ld[0][3], start_point))

        start_left_bb = tuple(sum(x) for x in zip(outs_ld[0][4], start_point))
        end_left_bb = tuple(sum(x) for x in zip(outs_ld[0][5], start_point))
        start_right_bb = tuple(sum(x) for x in zip(outs_ld[0][6], start_point))
        end_right_bb = tuple(sum(x) for x in zip(outs_ld[0][7], start_point)) 

       
        left_eye, right_eye= crop_eyes(frame.copy(), (start_left_bb, end_left_bb, start_right_bb, end_right_bb))

                
        # pitch, roll and yaw estimation on cropped face
        outs_hp , hp_inf_time= hp.predict(cropped_face.copy())  
        p, r, y = outs_hp
        
        # gaze estimation
        outs_gd, gd_inf_time = gd.predict(left_eye, right_eye, np.array([[y, p, r]]))

        # adding inference time to dictionary
        avg_inf_time["fd"].append(fd_inf_time)
        avg_inf_time["ld"].append(ld_inf_time)
        avg_inf_time["hp"].append(hp_inf_time)
        avg_inf_time["gd"].append(gd_inf_time)

        ## Control Mouse pointer 
        mc = MouseController("high", "fast")
        if len(outs_gd) == 0:
            continue

        mc.move(outs_gd[0], outs_gd[1])

        if len(visualization_list) != 0:
            show_visualization(frame, visualization_list, start_point, end_point, (start_left_bb, end_left_bb, start_right_bb, end_right_bb), [p1, p2, p3, p4], (p, r, y) ,outs_gd)

        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        

    logging.debug("Average inf. time are fd : {}, ld : {}, hp : {}, gd : {}".format(sum(avg_inf_time["fd"])/ len(avg_inf_time["fd"]), sum(avg_inf_time["ld"])/ len(avg_inf_time["ld"]), sum(avg_inf_time["hp"])/ len(avg_inf_time["hp"]) ,sum(avg_inf_time["gd"])/ len(avg_inf_time["gd"])))    
    logging.debug("Total inf. time are fd : {}, ld : {}, hp : {}, gd : {}".format(sum(avg_inf_time["fd"]), sum(avg_inf_time["ld"]), sum(avg_inf_time["hp"]) ,sum(avg_inf_time["gd"])))
    logging.debug("FPS time are fd : {}, ld : {}, hp : {}, gd : {}".format(1/(sum(avg_inf_time["fd"])/ len(avg_inf_time["fd"])), 1/(sum(avg_inf_time["ld"])/ len(avg_inf_time["ld"])), 1/(sum(avg_inf_time["hp"])/ len(avg_inf_time["hp"])) ,1 / (sum(avg_inf_time["gd"])/ len(avg_inf_time["gd"]))))
    logging.info("Stream Ended")
    cv2.destroyAllWindows()
    input_feed.close()
    

if __name__ == "__main__":
    main()