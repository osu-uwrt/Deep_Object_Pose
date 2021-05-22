import matplotlib.pyplot as plt
import csv
def plotLoss(fileName1, fileName2):
    # first file (train data)
    print()
    print('data set 1')
    with open(fileName1) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # line_count = 0
        
        #print(list(csv_reader))
        loss_data1 = list(csv_reader)

    # average epoch
    average_loss1 = []
    for i in range(120):
        losses1 = []
        for row in loss_data1[1:]:
            if int(row[0]) == i+1:
                losses1.append(float(row[2]))
        average1 = sum(losses1) / len(losses1)
        average_loss1.append(average1)
        print(average1)
    
    # second file (test data)
    print()
    print('data set 2')
    with open(fileName2) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # line_count = 0
        
        #print(list(csv_reader))
        loss_data2 = list(csv_reader)

    # average epoch
    average_loss2 = []
    for i in range(120):
        losses2 = []
        for row in loss_data2[1:]:
            if int(row[0]) == i+1:
                losses2.append(float(row[2]))
        average2 = sum(losses2) / len(losses2)
        average_loss2.append(average2)
        print(average2)
    

    # graph
    x1 = range(len(average_loss1))
    x2 = range(len(average_loss2))

    x_label = range(0,61,5)
    y = [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]


    plt.figure(figsize=(6.8, 4.2))
    plt.plot(x1,average_loss1, label='train')
    plt.plot(x2,average_loss2, label='test')
    plt.legend()

    plt.xticks(x_label, x_label)
    plt.yticks(y,y)
    plt.axis([0,60,0,0.005])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.suptitle('Loss for Train and Test')

    plt.savefig('/home/uwrt/Deep_Object_Pose/scripts/plot_png/loss_plot.png')
    print()
    print('check png')

    plt.show()
    


plotLoss('/mnt/Data/DOPE_trainings/train_cutie_04_18_2021/loss_train.csv', '/mnt/Data/DOPE_trainings/train_cutie_04_18_2021/loss_test.csv')
#print('done')