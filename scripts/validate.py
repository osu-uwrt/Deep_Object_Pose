
import sys
sys.path.append(".")

import cv2
from src.dope.inference.cuboid import Cuboid3d
from src.dope.inference.cuboid_pnp_solver import CuboidPNPSolver
from src.dope.inference.detector import ModelData, ObjectDetector
import numpy as np


def main():
    models = {}
    pnp_solvers = {}

    config_detect = lambda: None
    config_detect.mask_edges = 1
    config_detect.mask_faces = 1
    config_detect.vertex = 1
    config_detect.threshold = 0.5
    config_detect.softmax = 1000
    config_detect.thresh_angle = 0.5
    config_detect.thresh_map = 0.01
    config_detect.sigma = 3
    config_detect.thresh_points = 0.1

    weights = {
        # "cracker":"package://dope/weights/cracker_60.pth",
        # "gelatin":"package://dope/weights/gelatin_60.pth",
        # "meat":"package://dope/weights/meat_20.pth",
        # "mustard":"package://dope/weights/mustard_60.pth",
        "cutie":"/mnt/Data/DOPE_trainings/train_cutie_04_18_2021/net_cutie_100.pth",
        #"sugar":"package://dope/weights/sugar_60.pth",
        # "bleach":"package://dope/weights/bleach_28_dr.pth"
        
        # NEW OBJECTS - HOPE
        # "AlphabetSoup":"package://dope/weights/AlphabetSoup.pth", 
        # "BBQSauce":"package://dope/weights/BBQSauce.pth", 
        # "Butter":"package://dope/weights/Butter.pth", 
        # "Cherries":"package://dope/weights/Cherries.pth", 
        # "ChocolatePudding":"package://dope/weights/ChocolatePudding.pth", 
        # "Cookies":"package://dope/weights/Cookies.pth", 
        # "Corn":"package://dope/weights/Corn.pth", 
        # "CreamCheese":"package://dope/weights/CreamCheese.pth", 
        # "GreenBeans":"package://dope/weights/GreenBeans.pth", 
        # "GranolaBars":"package://dope/weights/GranolaBars.pth", 
        # "Ketchup":"package://dope/weights/Ketchup.pth", 
        # "MacaroniAndCheese":"package://dope/weights/MacaroniAndCheese.pth", 
        # "Mayo":"package://dope/weights/Mayo.pth", 
        # "Milk":"package://dope/weights/Milk.pth", 
        # "Mushrooms":"package://dope/weights/Mushrooms.pth", 
        # "Mustard":"package://dope/weights/Mustard.pth", 
        # "Parmesan":"package://dope/weights/Parmesan.pth", 
        # "PeasAndCarrots":"package://dope/weights/PeasAndCarrots.pth",
        # "Peaches":"package://dope/weights/Peaches.pth",
        # "Pineapple":"package://dope/weights/Pineapple.pth",
        # "Popcorn":"package://dope/weights/Popcorn.pth",
        # "OrangeJuice":"package://dope/weights/OrangeJuice.pth", 
        # "Raisins":"package://dope/weights/Raisins.pth",
        # "SaladDressing":"package://dope/weights/SaladDressing.pth",
        # "Spaghetti":"package://dope/weights/Spaghetti.pth",
        # "TomatoSauce":"package://dope/weights/TomatoSauce.pth",
        # "Tuna":"package://dope/weights/Tuna.pth",
        # "Yogurt":"package://dope/weights/Yogurt.pth",

    }

    dimensions = {
        "cracker": [16.403600692749023,21.343700408935547,7.179999828338623],
        "gelatin": [8.918299674987793, 7.311500072479248, 2.9983000755310059],
        "meat": [10.164673805236816,8.3542995452880859,5.7600898742675781],
        "mustard": [9.6024150848388672,19.130100250244141,5.824894905090332],
        "soup": [6.7659378051757813,10.185500144958496,6.771425724029541],
        "sugar": [9.267730712890625,17.625339508056641,4.5134143829345703],
        "bleach": [10.267730712890625,26.625339508056641,7.5134143829345703],
        "cutie": [100, 200, 1],

        # new objects
        "AlphabetSoup" : [ 8.3555002212524414, 7.1121001243591309, 6.6055998802185059 ], 
        "Butter" : [ 5.282599925994873, 2.3935999870300293, 10.330100059509277 ], 
        "Ketchup" : [ 14.860799789428711, 4.3368000984191895, 6.4513998031616211 ],     
        "Pineapple" : [ 5.7623000144958496, 6.95989990234375, 6.567500114440918 ],
        "BBQSauce" : [ 14.832900047302246, 4.3478999137878418, 6.4632000923156738 ], 
        "MacaroniAndCheese" : [ 16.625600814819336, 4.0180997848510742, 12.350899696350098 ], 
        "Popcorn" : [ 8.4976997375488281, 3.825200080871582, 12.649200439453125 ],
        "Mayo" : [ 14.790200233459473, 4.1030998229980469, 6.4541001319885254 ], 
        "Raisins" : [ 12.317500114440918, 3.9751999378204346, 8.5874996185302734 ],
        "Cherries" : [ 5.8038997650146484, 7.0907998085021973, 6.6101999282836914 ], 
        "Milk" : [ 19.035800933837891, 7.326200008392334, 7.2154998779296875 ], 
        "SaladDressing" : [ 14.744099617004395, 4.3695998191833496, 6.403900146484375 ],
        "ChocolatePudding" : [ 4.947199821472168, 2.9923000335693359, 8.3498001098632812 ], 
        "Mushrooms" : [ 3.3322000503540039, 7.079899787902832, 6.5869998931884766 ], 
        "Spaghetti" : [ 4.9836997985839844, 2.8492999076843262, 24.988100051879883 ],
        "Cookies" : [ 16.724300384521484, 4.015200138092041, 12.274600028991699 ], 
        "Mustard" : [ 16.004999160766602, 4.8573999404907227, 6.5132999420166016 ], 
        "TomatoSauce" : [ 8.2847003936767578, 7.0198001861572266, 6.6469998359680176 ],
        "Corn" : [ 5.8038997650146484, 7.0907998085021973, 6.6101999282836914 ], 
        "OrangeJuice" : [ 19.248300552368164, 7.2781000137329102, 7.1582999229431152 ], 
        "Tuna" : [ 3.2571001052856445, 7.0805997848510742, 6.5837001800537109 ],
        "CreamCheese" : [ 5.3206000328063965, 2.4230999946594238, 10.359000205993652 ], 
        "Parmesan" : [ 10.286199569702148, 6.6093001365661621, 7.1117000579833984 ], 
        "Yogurt" : [ 5.3677000999450684, 6.7961997985839844, 6.7915000915527344 ],
        "GranolaBars" : [ 12.400600433349609, 3.8738000392913818, 16.53380012512207 ], 
        "Peaches" : [ 5.7781000137329102, 7.0961999893188477, 6.5925998687744141 ],
        "GreenBeans" : [ 5.758699893951416, 7.0608000755310059, 6.5732002258300781 ], 
        "PeasAndCarrots" : [ 5.8512001037597656, 7.0636000633239746, 6.5918002128601074 ] 
    }

    # For each object to detect, load network model, create PNP solver, and start ROS publishers
    for model, weights_url in weights.items():
        models[model] = \
            ModelData(
                model,
                weights_url
            )
        models[model].load_net_model()



        pnp_solvers[model] = \
            CuboidPNPSolver(
                model,
                cuboid3d=Cuboid3d(dimensions[model])
            )

        camera_matrix = np.array([[618,    0,         256.0],
                                [  0,      618,       256.0],
                                [  0,      0.,        1.        ]])
        dist_coeffs = np.array([[0.],
                        [0.],
                        [0.],
                        [0.]])

        

    # read the image(jpg) on which the network should be tested. 
    # example: 
    # C:\\Users\\m\\Desktop\\000044.jpg
    

    for i in range(20):
        pathToImg = "/mnt/Data/visii_data/cutie/cutie_training/cutie%d.png" % i
        print("path to the image is: {}".format(pathToImg))
        img = cv2.imread(pathToImg)
        if img is None:
            continue
        cv2.imshow('img', img)
        cv2.waitKey(1)

        height, width, _ = img.shape
        scaling_factor = float(400) / height
        if scaling_factor < 1.0:
            img = cv2.resize(img, (int(scaling_factor * width), int(scaling_factor * height)))
            camera_matrix *= scaling_factor
        
        pnp_solvers[model].set_camera_intrinsic_matrix(camera_matrix)
        pnp_solvers[model].set_dist_coeffs(dist_coeffs)
        
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
    main()