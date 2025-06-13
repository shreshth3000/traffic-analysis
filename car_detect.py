from ultralytics import YOLO
import cv2 as cv

focal_length=868.2131
orig_size=24


model = YOLO('./models/yolo8m.pt')




vid=cv.VideoCapture("vid.mp4")
vid.set(cv.CAP_PROP_FRAME_WIDTH,1920)
vid.set(cv.CAP_PROP_FRAME_HEIGHT,1080)
vid.set(cv.CAP_PROP_FPS,60)

desired_obj=[0,3]

while True:
    istrue,frame=vid.read()
    frame=cv.resize(frame,(1220,700))

    resize_frame=cv.resize(frame,(640,640))
    rgb_frame=cv.cvtColor(resize_frame,cv.COLOR_BGR2RGB)
    results=model(rgb_frame,device="cpu")
    scale_x = frame.shape[1] / 640
    scale_y = frame.shape[0] / 640
    if results is None:
        continue
    
    else:
        for result in results:
            for box in result.boxes:
                if box.cls in desired_obj:
                    x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
            

                    cv.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),thickness=4)
                        # cv.putText(frame,str((focal_length*orig_size)/(y2-y1)),(x1,y1-100),fontFace=cv.FONT_HERSHEY_SIMPLEX,thickness=3,fontScale=1,color=(255,0,0))
    cv.imshow("vid",frame)

    if cv.waitKey(10) & 0xFF==ord("d"):

        break

cv.destroyAllWindows()
vid.release()


