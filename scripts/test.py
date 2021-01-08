#!/usr/bin/env python

import cv2
from dope.inference.cuboid import Cuboid3d
from dope.inference.cuboid_pnp_solver import CuboidPNPSolver
from dope.inference.detector import ModelData, ObjectDetector
import numpy as np
import rospy
import resource_retriever
import tf.transformations


def main():
    models = {}
    pnp_solvers = {}

    config_detect = lambda: None
    config_detect.mask_edges = 1
    config_detect.mask_faces = 1
    config_detect.vertex = 1
    config_detect.threshold = 0.5
    config_detect.softmax = 1000
    config_detect.thresh_angle = rospy.get_param('~thresh_angle', 0.5)
    config_detect.thresh_map = rospy.get_param('~thresh_map', 0.01)
    config_detect.sigma = rospy.get_param('~sigma', 3)
    config_detect.thresh_points = rospy.get_param("~thresh_points", 0.1)

    # For each object to detect, load network model, create PNP solver, and start ROS publishers
    for model, weights_url in rospy.get_param('~weights').iteritems():
        models[model] = \
            ModelData(
                model,
                resource_retriever.get_filename(weights_url, use_protocol=False)
            )
        models[model].load_net_model()



        pnp_solvers[model] = \
            CuboidPNPSolver(
                model,
                cuboid3d=Cuboid3d(rospy.get_param('~dimensions')[model])
            )

        camera_matrix = np.array([[205.2789348,    0,         355.83333333],
                    [  0,         205.2789348,  200.27777778],
                    [  0,           0.,           1.        ]])
        dist_coeffs = np.array([[0.],
                        [0.],
                        [0.],
                        [0.]])

        pnp_solvers[model].set_camera_intrinsic_matrix(camera_matrix)
        pnp_solvers[model].set_dist_coeffs(dist_coeffs)

    # read the image(jpg) on which the network should be tested. 
    # example: 
    # C:\\Users\\m\\Desktop\\000044.jpg
    pathToImg = "/home/blaine/Pictures/net_soup.png"
    print("path to the image is: {}".format(pathToImg))
    img = cv2.imread(pathToImg)
    cv2.imshow('img', img)
    cv2.waitKey(1)

    height, width, _ = img.shape
    scaling_factor = float(400) / height
    if scaling_factor < 1.0:
        img = cv2.resize(img, (int(scaling_factor * width), int(scaling_factor * height)))


    for m in models:
        # try to detect object
        results, im_belief = ObjectDetector.detect_object_in_image(models[m].net, pnp_solvers[m], img, config_detect, grid_belief_debug=True, norm_belief=True, run_sampling=True)

        print("objects found: {}".format(results))
        cv_imageBelief = np.array(im_belief)
        imageToShow = cv2.resize(cv_imageBelief, dsize=(800, 800))
        cv2.imshow('beliefMaps', imageToShow)
        cv2.waitKey(0)      

    print("end")


if __name__ == '__main__':
    rospy.init_node('dope')
    main()
    rospy.spin()