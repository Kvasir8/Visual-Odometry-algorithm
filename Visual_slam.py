import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

import SuperPointPretrainedNetwork.demo_superpoint as SPPN
# import demo_superpoint as SPNNN

# from SuperPointPretrainedNetwork.demo_superpoint import opt # from python script.py import opt object


def nn_match_two_way(desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches


#https://python.hotexamples.com/examples/cv2/-/findEssentialMat/python-findessentialmat-function-examples.html
#https://www.programcreek.com/python/example/110761/cv2.findEssentialMat
def estimate_relative_pose_from_correspondence(pts1, pts2, K1, K2):
    f_avg = (K1[0, 0] + K2[0, 0]) / 2
    pts1, pts2 = np.ascontiguousarray(pts1, np.float32), np.ascontiguousarray(pts2, np.float32)

    pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K1, distCoeffs=None)
    pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K2, distCoeffs=None)

    E_, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.),
                                   method=cv2.RANSAC, prob=0.999, threshold=3.0 / f_avg)
    points, R_est, t_est, mask_pose = cv2.recoverPose(E_, pts_l_norm, pts_r_norm)
    # return mask_pose[:, 0].astype(np.bool), R_est, t_est
    return mask_pose, R_est, t_est


#https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
def epilines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


#HINT: Needed for Pose Graph Optimization
# from posegraphoptimizer import PoseGraphOptimizer, getGraphNodePose

# Util function (returning transformation matrix 4x4)
def T_from_R_t(R, t):
    R = np.array(R).reshape(3, 3)
    t = np.array(t).reshape(3)
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[0, 3] = t[0]
    T[1, 3] = t[1]
    T[2, 3] = t[2]
    T[3, 3] = 1
    return T

#Tracking draw: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
def draw_feature_tracked(second_frame, first_frame,
                        second_keypoints, first_keypoints,
                        color_line=(0, 255, 0), color_circle=(255, 0, 0)):
    mask_bgr = np.zeros_like(cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR))
    frame_bgr = cv2.cvtColor(second_frame, cv2.COLOR_GRAY2BGR)
    for i, (second, first) in enumerate(zip(second_keypoints, first_keypoints)):
        a, b = second.ravel()
        c, d = first.ravel()
        mask_bgr = cv2.line(mask_bgr, (int(a), int(b)), (int(c), int(d)), color_line, 1)
        frame_bgr = cv2.circle(frame_bgr, (int(a), int(b)), 3, color_circle, 1)
    return cv2.add(frame_bgr, mask_bgr)


def getGT(file_context, frame_id):
    ss = file_context[frame_id].strip().split()
    x = float(ss[3])
    y = float(ss[7])
    z = float(ss[11])
    return [x, y, z]


class Camera:
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


# Major functions for VO computation
class VO:
    def __init__(self, camera):
        self.camera = camera
        self.focal = self.camera.fx
        self.center = (self.camera.cx, self.camera.cy)

        self.curr_R = None
        self.curr_t = None

        self.T = None
        self.relative_T = None
        #https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_py_fast.html
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

        self.K_ = np.zeros((3,3))

    def featureTracking(self, curr_frame, old_frame, old_kps):
        # ToDo
        ## Not: There is a optical flow method in OpenCV that can help ;) input the old_kps and track them

        lk_params = dict(winSize=(15, 15),
                         maxLevel=2, # more deeper level can find faster movement in optical flow
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        curr_kps, matches, err = cv2.calcOpticalFlowPyrLK(old_frame, curr_frame, old_kps, None, **lk_params)
        # old_frame = curr_frame.copy()
        # old_kps = curr_kps.copy()
        # old_kps = curr_kps
        # while (1):
            # ret, frame = cap.read()
            # if not ret:
            #     print('No frames grabbed!')
            #     break
            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) #[hkh] no need to convert as the source image is already gray

            ## calculate optical flow
            # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_kps, None, **lk_params)
            # curr_kps, matches, err = cv2.calcOpticalFlowPyrLK(old_frame, curr_frame, old_kps, None, **lk_params)
            # old_frame = curr_frame.copy()
            # old_kps = curr_kps.copy()
            # x, y = curr_kps.ravel()

            ## Select good points
            # if p1 is not None:
            #     good_new = p1[st == 1]
            #     good_old = p0[st == 1]
            ## draw the tracks
            # for i, (new, old) in enumerate(zip(good_new, good_old)):
            #     a, b = new.ravel()
            #     c, d = old.ravel()
            #     mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            #     frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            # img = cv2.add(frame, mask)
            #
            # cv2.imshow('frame', img)
            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
            #     break
            ## Now update the previous frame and previous points
            # old_gray = frame_gray.copy()
            # p0 = good_new.reshape(-1, 1, 2)
        # cv2.destroyAllWindows()

        ###
        # Remove nono-matched keypoints
        matches = matches.reshape(matches.shape[0])
        return curr_kps[matches == 1], old_kps[matches == 1], matches

    # def featureMatching(self, curr_frame, old_frame, orb=True):
    def featureMatching(self, curr_frame, old_frame, orb=False, sp=False):
        if orb:
            # ToDo
            matches = []
            # Hint: again, OpenCV is your friend ;) Tip: maybe you want to improve the feature matching by only taking the best matches...
            #https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_py_matcher.html
            img1 = curr_frame # queryImage
            img2 = old_frame # trainImage

            # Initiate ORB detector
            orb = cv2.ORB_create()
            # find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(img1,None)
            kp2, des2 = orb.detectAndCompute(img2,None)

            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(des1, des2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            # Finally draw the lines representing the equalities: first 10 matches.
            Mtch_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, flags=2, outImg=None)

            plt.imshow(Mtch_img), plt.show()


        # [hkh]
        elif sp:
            # os.system("SuperPointPretrainedNetwork/demo_superpoint.py assets/icl_snippet/")
            # opt = SPPN.opt
            first_frame = cv2.normalize(old_frame, None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # normalize
            first_frame = first_frame.astype(np.float32)
            second_frame = cv2.normalize(curr_frame, None, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # normalize
            second_frame = second_frame.astype(np.float32)
            # following init hyperparams from demo_superpoint.py
            fe = SPPN.SuperPointFrontend(weights_path='superpoint_v1.pth',
                                         nms_dist=4,
                                         conf_thresh=0.015,
                                         nn_thresh=0.7,
                                         cuda=False)
            print('__Successfully loaded pre-trained network__')
            # This class helps merge consecutive point matches into tracks.
            tracker = SPPN.PointTracker(max_length=5, nn_thresh=fe.nn_thresh)

            pts1, desc_1, _ = fe.run(img=first_frame)
            pts2, desc_2, _ = fe.run(img=second_frame)
            first_keypoints = np.array(pts1[:2, :])
            second_keypoints = np.array(pts2[:2, :])
            print('__loaded fe and pts__')
            # matches = SPPN.PointTracker.nn_match_two_way(desc1=desc_1, desc2=desc_2, nn_thresh=0.7)
            matches = nn_match_two_way(desc1=desc_1, desc2=desc_2, nn_thresh=0.7)
            print("__matched retrieved__")

            second_keypoints_matched = first_keypoints[:2, matches.astype(int)[0]].transpose()
            first_keypoints_matched = second_keypoints[:2, matches.astype(int)[1]].transpose()

            # kp1 = first_keypoints[:2, matches.astype(int)[0]].transpose()
            # kp2 = second_keypoints[:2, matches.astype(int)[1]].transpose()

            return first_keypoints_matched, second_keypoints_matched, matches

            ###
        else:  # use SIFT
            # ToDo
            # Hint: Have you heard about the Ratio Test for SIFT?
            #[hkh]KPmatching: ratio of closest-dist to 2nd-closest dist is taken
            #[hkh]Ratio test:https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
            img1 = curr_frame # queryImage
            img2 = old_frame # trainImage

            sift = cv2.SIFT_create()

            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(len(matches))]
            # matchesMask = [0 for i in range(len(matches))]

            ListIdx = [] # List indices where the accepted Kps are located.
            matches_list = []
            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance: # distance ratio between 1st and 2nd best match, accept if 1st best match (m) is better
                    # by reaching certain threshold, 0.7: 2nd best one is larger than k=1 1st best Kp match (m) is
                    matchesMask[i] = [1, 0]
                    ListIdx.append(i) #[hkh]

            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=0)

            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

            plt.imshow(img3, ), plt.show()

            #[hkh]
            for i in ListIdx:
                matches_list.append(matches[i])
            matches = np.array(matches_list)
            matches = matches[:,0]

            #[hkh]matches[mat][matchesMask].queryIdx.pt
            # matches = np.array(matches)         #Make Numpy array first to pass the argument
            # matches = matches[matchesMask]      #pass argument and get desired array, (3361,)
            # matchesMask.resize((len(matchesMask)), refcheck=False)

        # elif sp: # [hkh] SuperPoint
        #     SPPN.PointTracker.nn_match_two_way()

            ###
        kp1_match = np.array([kp1[mat.queryIdx].pt for mat in matches])
        kp2_match = np.array([kp2[mat.trainIdx].pt for mat in matches])

        return kp1_match, kp2_match, matches

    # def initialize(self, first_frame, second_frame, of=True, orb=False):
    def initialize(self, first_frame, second_frame, of=True, orb=False, sp=False): #[hkh]SP
        if of:
            first_keypoints = self.detector.detect(first_frame)
            first_keypoints = np.array([x.pt for x in first_keypoints], dtype=np.float32)
            second_keypoints_matched, first_keypoints_matched, _ = self.featureTracking(second_frame, first_frame,
                                                                                        first_keypoints)

        else:
        # elif orb: #[hkh]
            second_keypoints_matched, first_keypoints_matched, _ = self.featureMatching(second_frame, first_frame,
                                                                                        orb=orb, sp=sp)

        # ToDo
        # Hint: Remember the lecture: given the matched keypoints you can compute the Essential matrix and from E you can recover R and t...
        #c.f. cv2 chpt5 p.10
        #[hkh]Fundamental Matrix: https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
        #[hkh]Now extract R, T from E
        self.K_ = np.array([ [self.camera.fx, 0, self.camera.cx], [0, self.camera.fy, self.camera.cy], [0, 0, 1] ])
        # K_ = np.vstack((K_23, np.array([0, 0, 1])))
        _, self.curr_R, self.curr_t = estimate_relative_pose_from_correspondence(first_keypoints_matched, second_keypoints_matched, self.K_, self.K_)

        ###

        self.relative_T = T_from_R_t(self.curr_R, self.curr_t)
        self.T = self.relative_T
        return second_keypoints_matched, first_keypoints_matched

    # def processFrame(self, curr_frame, old_frame, old_kps, of=True, orb=False):
    def processFrame(self, curr_frame, old_frame, old_kps, of=True, orb=False, sp=False): #[hkh]

        if of: #Optical Flow
            curr_kps_matched, old_kps_matched, matches = self.featureTracking(curr_frame, old_frame,
                                                                                           old_kps)
        # elif sp: #[hkh]
        #     curr_kps_matched, old_kps_matched, matches = self.featureMatching(curr_frame, old_frame,
        #                                                                                    sp=sp)
        else:
        # elif orb: #[hkh]
            curr_kps_matched, old_kps_matched, matches = self.featureMatching(curr_frame, old_frame,
                                                                                           orb=orb, sp=sp)


        # ToDo
        # Hint: Here we only do the naive way and do everything based on Epipolar Geometry (Essential Matrix). No need for PnP in this tutorial
        #https://python.hotexamples.com/examples/cv2/-/findEssentialMat/python-findessentialmat-function-examples.html
        mask, R, t = estimate_relative_pose_from_correspondence(curr_kps_matched,old_kps_matched,
                                                                                    self.K_, self.K_)

        ###
        inliners = len(mask[mask == 255])
        if (inliners > 20):
            self.relative_T = T_from_R_t(R, t)
            self.curr_t = self.curr_t + self.curr_R.dot(t)
            self.curr_R = R.dot(self.curr_R)
            self.T = T_from_R_t(self.curr_R, self.curr_t)

        # Get new KPs if too few
        if (old_kps_matched.shape[0] < 1000):
            curr_kps_matched = self.detector.detect(curr_frame)
            curr_kps_matched = np.array([x.pt for x in curr_kps_matched], dtype=np.float32)
        return curr_kps_matched, old_kps_matched

def main():
    argument = argparse.ArgumentParser()
    argument.add_argument("--o", help="use ORB", action="store_true")
    argument.add_argument("--sp", help="use SuperPoint", action="store_true") #[hkh]
    argument.add_argument("--f", help="use Optical Flow", action="store_true")
    argument.add_argument("--l", help="use Loop Closure for PGO", action="store_true")
    args = argument.parse_args()
    orb = args.o
    of = args.f
    sp = args.sp #[hkh]
    loop_closure = args.l

    #Hard-coded Loop closure estimates (Needed for PGO); We only take these 2 for now
    lc_ids = [1572, 3529]
    lc_dict = {1572: 125, 3529: 553}

    #HINT: Adapt path
    image_dir = os.path.realpath("../../dataset/kitti/00/image_0/")
    pose_path = os.path.realpath("../../dataset/kitti/poses/00.txt")

    with open(pose_path) as f:
        poses_context = f.readlines()

    image_list = []
    for file in os.listdir(image_dir):
        if file.endswith("png"):
            image_list.append(image_dir + '/' + file)

    image_list.sort()

    # Initial VisualOdometry Object (=Camera intrinsic matrix)
    camera = Camera(1241.0, 376.0, 718.8560,
                    718.8560, 607.1928, 185.2157)
    vo = VO(camera)
    traj_plot = np.zeros((1000,1000,3), dtype=np.uint8)

    # ToDo (PGO)
    #Hint: Initialize Pose Graph Optimizer
    # Hint: have a look in the PGO class and what methods are provided. The first frame should be static (addPriorFactor)

    ###

    first = 0
    second = first + 3  # For wider baseline with better initialization...
    first_frame = cv2.imread(image_list[first], 0)
    second_frame = cv2.imread(image_list[second], 0)

    #[hkh]
    second_keypoints, first_keypoints = vo.initialize(first_frame, second_frame, of=of, orb=orb, sp=sp)

    # ToDo (PGO)
    # Hint: fill the Pose Graph: There is a difference between the absolute pose and the relative pose



    ###


    old_frame = second_frame
    old_kps = second_keypoints


    for index in range(second+1, len(image_list)):
        curr_frame = cv2.imread(image_list[index], 0)
        true_pose = getGT(poses_context, index)
        true_x, true_y = int(true_pose[0])+290, int(true_pose[2])+90

        # curr_kps, old_kps = vo.processFrame(curr_frame, old_frame, old_kps, of=of, orb=orb)
        curr_kps, old_kps = vo.processFrame(curr_frame, old_frame, old_kps, of=of, orb=orb, sp=sp) #[hkh]

        # ToDo (PGO)
        # Hint: keep filling new poses



        ###

        if loop_closure:
            if index in lc_ids:
                loop_idx = lc_dict[index]
                print("Loop: ", PGO.curr_node_idx, loop_idx)

                # ToDo (PGO)
                # Hint: just use Identity pose for Loop Closure np.eye(4)



                ###

                #Plot trajectory after PGO
                for k in range(index):
                    try:
                        pose_trans, pose_rot = getGraphNodePose(PGO.graph_optimized, k)
                        print(pose_trans)
                        print(pose_rot)
                        cv2.circle(traj_plot, (int(pose_trans[0])+290, int(pose_trans[2])+90), 1, (255, 0, 255), 5)
                    except:
                        #catch error for first few missing poses...
                        print("Pose not available for frame # ", k)


        #Utilities for Drawing
        curr_t = vo.curr_t
        if(index > 2):
            x, y, z = curr_t[0], curr_t[1], curr_t[2]
        else:
            x, y, z = 0., 0., 0.
        odom_x, odom_y = int(x)+290, int(z)+90

        cv2.circle(traj_plot, (odom_x,odom_y), 1, (index*255/4540,255-index*255/4540,0), 1)
        cv2.circle(traj_plot, (true_x,true_y), 1, (0,0,255), 2)
        cv2.rectangle(traj_plot, (10, 20), (600, 60), (0,0,0), -1)
        text = "FrameID: %d  Coordinates: x=%1fm y=%1fm z=%1fm"%(index,x,y,z)
        cv2.putText(traj_plot, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
        cv2.imshow('Trajectory', traj_plot)
        show_image = draw_feature_tracked(curr_frame, old_frame,
                                         curr_kps, old_kps)
        cv2.imshow('Mono', show_image)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break



        # Update old data
        old_frame = curr_frame
        old_kps = curr_kps

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
