def extract_face(in_img,out_img,resize):
    # Import
    import os
    import cv2
    
    # SETTINGS
    print('Input image =' ,in_img)
    color = (255,255,255)

    cascade  = cv2.CascadeClassifier(".\\haarcascade_frontalface_default.xml")

    if os.path.exists(in_img):
        print("Input image exists!")
        # import img
        image = cv2.imread( str(in_img) )
        image_gs = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # extract face and resize
        face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(resize, resize))

        if len(face_list) >0: # when one or more face extracted
            rect = face_list[0] # 1st Element Only
            x,y,width,height=rect
            image = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]] 
            if image.shape[0]> resize:
                image = cv2.resize(image,(resize,resize))

                #Save Image
                cv2.imwrite(out_img,image)
        else:
            print("Face Not Found")
    else:
        print("Image Not Found")
        