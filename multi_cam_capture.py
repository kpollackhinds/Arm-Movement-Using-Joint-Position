import cv2 
import threading

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    
    def run(self):
        print (f"Starting {self.previewName}")
        self.camPreview(self.previewName, self.camID)
    
    def camPreview(previewName, camID):
        cv2.namedWindow(previewName)
        cam = cv2.VideoCapture(camID)
        
        if cam.isOpened():
            ret, frame = cam.read()
        else:
            ret = False

        while ret:
            ret, frame = cam.read()
            cv2.imshow(previewName, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow(previewName)
        cam.release()

    
