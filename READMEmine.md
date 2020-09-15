Face-detection

In the field of security, motion detection remains a research area despite the evolution it has known since its creation. A surveillance camera can recover any type of movement. The movement that interests us in this project is the movement of human beings. The camera should only detect the movement of human beings. This detection is based on facial recognition techniques.

Evaluate the trained model using:
```Shell
# dataset choices = ['AFW', 'PASCAL', 'FDDB','testmybdd']
python3 test.py --dataset testmybdd
# evaluate using cpu
python3 test.py --cpu
# visualize detection results
python3 test.py -s --vis_thres 0.3
#run the MyTest.py with video input
$ python MyTest.py --video samples/vid.mp4 --output-dir outputs/
#run the MyTest.py in your own webcam
$ python MyTest.py --src 0 --output-dir outputs/
#Test phase using surveillance video use
python testVideo.py --video samples/vidd.avi 

#To run the code on the HPC just change the line in script.sh based on the need
python train.py
             # in case of training
python test.py --dataset testmybdd  
             # in case of testing
