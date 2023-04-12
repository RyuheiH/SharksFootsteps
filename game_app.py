import cv2
import time
import glob
import numpy as np
import pygame.mixer
from PIL import Image, ImageDraw, ImageFont


"""
There are 6 displays in general, and from main() function, 5 of them can be shown. 
Those startpage(), play_game(), show_page(), show_screenshot(), functions will show a video or a screen, so each page in the game runs with each function.
play_the_shark_attack() is called from play_game() when its game over.
"""


def main(): #this main is in charge of showing the displays, and after pressing Q in those displays, they will come back here.

    pygame.mixer.init() #initial music function
    while True:
        pygame.mixer.music.load("music/氷雨.mp3") # load bgm
        enter_sound = pygame.mixer.Sound("music/決定ボタンを押す8.mp3")  #load SE
        pygame.mixer.music.play(1) #-1 means infinite loop
        key = startpage()
        if key == 13: #13 stands for enter key
            enter_sound.play()  #play SE
            play_game() #play_game function is in charge of showing the main game

        elif key == ord('h'):
            cv2.destroyAllWindows()
            show_page("How_to_play")
             
        elif key == ord('q'): #when q is pressed, it quits the while loop
            print("quitting")
            break

        elif key == ord('s'):
            cv2.destroyAllWindows()
            show_screenshot()

        elif key == ord('c'):
            cv2.destroyAllWindows()
            show_page("Credits")

        else:
            print("Not assigned key was pressed. Quitting...")
            break




def show_page(videoname):
    cap = cv2.VideoCapture(f"videos/{videoname}.mp4") #video of Startpage

    while True:
        ret, img = cap.read()

        if ret == False: #when the video is done
            cap.set(cv2.CAP_PROP_POS_FRAMES,0) #it sets back the video
            ret, img = cap.read() #and reads it again

        cv2.imshow(videoname, img) #show img
        key = cv2.waitKey(1) & 0xFF #waits for the key command

        if key == ord('q'): #when q is pressed, it quits the while loop
            cv2.destroyAllWindows()
            break
        elif key == 13: #it quits with enter key as well
            cv2.destroyAllWindows()



def show_screenshot():
    files = glob.glob("screenshot/capture_*.png") #read the files under screenshot folder
    n = 0

    for n in files:
        img = cv2.imread(n)
        text = "Press Enter to see other pictures"
        img = cv2_putText_5(img = img,text = text,position = [2,3],fontScale = 30,color = (255, 255, 255),mode = 2)
        
        cv2.imshow(f'{n}',img)
        

        
        key = cv2.waitKey(0) & 0xFF #waits for the key command
        if key == ord('q'): #when q is pressed, it quits the while loop
            cv2.destroyAllWindows()
            break
        elif key == 13: #it quits with enter key as well
            cv2.destroyAllWindows()



def startpage():
    capStart = cv2.VideoCapture("videos/StartPage.mp4") #video of Startpage
    while True:
        retStart, imgStart = capStart.read()

        if retStart == False: #when the shark video is done
            capStart.set(cv2.CAP_PROP_POS_FRAMES,0) #it sets back the video
            retStart, imgStart = capStart.read() #and reads it again

        cv2.imshow('Shark', imgStart) #show img
        key = cv2.waitKey(1) & 0xFF #waits for the key command

        if key != 255: #255 is default when not pressed anything.
            break
        
    return key



def play_game():
        
        cap = cv2.VideoCapture(0) #video from camera
        capShark = cv2.VideoCapture("videos/shark.mp4") #video of Shark
        imgDH = cv2.imread('graphic/imgDH_180.png',-1) #reading the image of diving helmet, with alpha layer
        smile_count = 0 #counts the times of smiles
        smilewait = [] #this array keeps the time of when it detected smiles, this is used to wait for making each screenshot
        face_detector = cv2.FaceDetectorYN.create("recognition_trained_data/yunet.onnx", "", (960,510)) #this is a face detector and the model is called YuNet, the numbers behind is the size of input graphic.
        cascade_smile = cv2.CascadeClassifier("recognition_trained_data/haarcascade_smile.xml") #using cascade to detect smile



        pygame.mixer.music.load("music/水中流星群.mp3") #bgm 
        pygame.mixer.music.play(-1) #-1 means infinite loop
        pygame.mixer.music.set_volume(0.3) #lower the volume to match the one from the startpage

        while True:

            ret, frame = cap.read()
            img = cv2.resize(frame,(960,510)) #resizing the size of video graphic for faster computing
            img = cv2.flip(img, 1) #mirror the camera
            retS, imgS = capShark.read()
            frame_num = capShark.get(cv2.CAP_PROP_POS_FRAMES) #getting which frame it is now, this is used to judge if the player is seen by the Shark or not

            if retS == False: #when the shark video is done
                capShark.set(cv2.CAP_PROP_POS_FRAMES,0) #it sets back the video
                retS, imgS = capShark.read() #and reads it again


            faces = facial_recognition(face_detector,img) #get faces
            if check_if_shark(faces,frame_num) is True: #if Shark sees it 
                play_the_shark_attack(face_detector,imgDH,cascade_smile,smile_count,smilewait) #play the video of Shark attacking 
                break
                

            for i, face in enumerate(faces): #it does trimming, emerging images, smile recognition for each face
                if i > 5 : break #the display can only put 5 people on the screen so if its bigger than that, it should give up

                trim = trim_the_detected_face(face,img) #trims the face 
                imgS = put_displays_all_together(imgDH,imgS,trim,i) #so that it can be in the helmet, and put in the main graphic
                smiles = smile_recognition(img,cascade_smile,face) #from the trimmed face it checks smile

                if len(smiles) > 0 : #if smile(s) are detected
                    smile_count, smilewait = smile_screenshot(smile_count,smilewait,imgS) #it takes a screenshot
                    if time.time() - smilewait[smile_count-1] < 2: #if this smile is detected, it puts the text for 2 seconds
                        imgS = cv2_putText_5(img = imgS,text = "Took Screenshot!",position=[2,2],fontScale = 60,color = (255, 255, 255),mode = 2)
                        


            cv2.imshow('Shark',imgS)

            key = cv2.waitKey(1) & 0xFF #waits for the key command
            if key == ord('q'): #when q is pressed, it quits the while loop
                break


        cap.release()
        cv2.destroyAllWindows()

 

def check_if_shark(faces,frame_num): #this function checks if the player is seen based on which frames the shark is in the screen
    if len(faces) > 0:
        if frame_num > 280 and frame_num < 380: #from frame number 280 to 380, Shark will see the player
            return True
        elif frame_num > 1020 and frame_num < 1150: #from frame number 1020 to 1150, Shark will see the player
            return True
    else :
        return False

                   



def play_the_shark_attack(face_detector,imgDH,cascade_smile,smile_count,smilewait): #this function is based on play_game but with less definition of variables since it is in the argument

    capAttack = cv2.VideoCapture("videos/shark_attack.mp4") #get the shark attacking video
    cap = cv2.VideoCapture(0) #video from camera

    
    pygame.mixer.music.load("music/se_bukimi2-1.mp3") #bgm 
    pygame.mixer.music.play(0) #-1 means infinite loop
    pygame.mixer.music.set_volume(0.2) #this scary sound is set to lower volume 


    while True:

        retA, imgA = capAttack.read()
        ret, frame = cap.read()
        img = cv2.resize(frame,(960,510)) #resize the 
        img = cv2.flip(img, 1) #mirror the camera to make it like a mirror
        
        if retA == False: #when the shark video is done
            break

        faces = facial_recognition(face_detector,img)


        for i, face in enumerate(faces):
            if i > 5 : break #the display can only put 5 people on the screen so if its bigger than that, it should give up

            trim = trim_the_detected_face(face,img)
            imgA = put_displays_all_together(imgDH,imgA,trim,i)
            smiles = smile_recognition(img,cascade_smile,face)

            if len(smiles) > 0 :
                smile_count, smilewait = smile_screenshot(smile_count,smilewait,imgA)
                if time.time() - smilewait[smile_count-1] < 2:
                    imgA = cv2_putText_5(img = imgA,text = "Took Screenshot!",position=[2,2],fontScale = 60,color = (255, 255, 255),mode = 2)
        

        imgA = cv2_putText_5(img = imgA,text = "Found By Shark!",position = [2,4],fontScale = 60,color = (85, 84, 234),mode = 2) #RGBではなくBGR234, 84, 85)
    
        
        cv2.imshow('GameOver',imgA)

        key = cv2.waitKey(1) & 0xFF #waits for the key command
        if key == ord('q'): #when q is pressed, it quits the while loop
            break

    capAttack.release()
    cv2.destroyAllWindows()
     



def trim_the_detected_face(face,img): #this function trims the detected face
            
    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3]) #x and y coordinates of the face, and width and height

    trim = img[y:y+h,x:x+w]   # range of x and y of face rectangle 
    
    if np.any(trim) :
        trim = cv2.resize(trim,(100,100))  #resize the window for the face
        
    return trim



def smile_screenshot(smile_count,smilewait,imgS): #this takes screenshot, and prevents from taking screenshots many times by using time


    if smile_count == 0: #when its the first smile, it just takes the screenshot
        smilewait.append(time.time())
        smile_count += 1
        cv2.imwrite(f"screenshot/capture_{smile_count}.png", imgS) #when smile is detected, takes screenshot
        #print("smiled")


    if time.time() - smilewait[smile_count-1] > 3 and smile_count > 0: #when it is not first time of screenshot, it cannot take screenshot until 3 seconds passes since the last screenshot
        smilewait.append(time.time())
        smile_count += 1
        cv2.imwrite(f"screenshot/capture_{smile_count}.png", imgS) #when smile is detected, takes screenshot
        #print("smiled")



    return smile_count, smilewait



def facial_recognition(face_detector,img): #facial recognition
        
    _, faces = face_detector.detect(img)

    faces = faces if faces is not None else [] #to avoid errors, if face is not recognized, it will make an empty array

    return faces



def smile_recognition(img,cascade_smile,face):
            
    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3]) #x and y coordinates of the face, and width and height
    roi = img[y:y+h, x:x+w] #for the smile recognition
    
    if np.any(roi):
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #gray scale video for recognition
        smiles = cascade_smile.detectMultiScale(roi_gray,scaleFactor= 1.2, minNeighbors=30, minSize=(100, 100)) #smile recognition
        return smiles
    
    smiles = smiles if smiles is not None else [] #to avoid errors, if smile is not recognized, it will make an empty array

    return smiles



def put_displays_all_together(imgDH,imgS,trim,i):

    # setting the roi (left(x1), top(y1), right(x2), bottom(y2))
    #this roi is used for putting the diving helmet, and it can show up to 5 people, decided by i
    roi_for_DH = [[0,20,180,200],[180,20,360,200],[360,20,540,200],[540,20,720,200],[720,20,900,200]]

    x_offset = [40,220,400,580] #offset to change the location of trimmed face
    y_offset = [40,40,40,40] #same height 

    h, w = trim.shape[:2] #trim is the trimmed face
    cut = 3/2 #some parts of face will be hidden by the diving helet so it will crop more
    trim_cropped = trim[0:round(h/cut), 0:round(w), :] #trimmed face is cropped

    imgS[y_offset[i]:y_offset[i]+trim_cropped.shape[0], x_offset[i]:x_offset[i]+trim_cropped.shape[1]] = trim_cropped #putting the face on the screen

    s_roi = imgS[roi_for_DH[i][1]: roi_for_DH[i][3], roi_for_DH[i][0]: roi_for_DH[i][2]] #the shark video has roi now(i is the count of ppl)
    s_roi = merge_images(s_roi,imgDH,0,0) #putting the diving helmet over it
    imgS[roi_for_DH[i][1]: roi_for_DH[i][3], roi_for_DH[i][0]: roi_for_DH[i][2]] = s_roi #putting the s_roi over the main image

    return imgS #return the merged image
     

def merge_images(bg, fg_alpha, s_x, s_y): #the sizes of two iamges have to be the same
    alpha = fg_alpha[:,:,3]  # get alpha channel
    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) # gray to BGR
    alpha = alpha / 255.0    # 0.0〜1.0

    fg = fg_alpha[:,:,:3]

    f_h, f_w, _ = fg.shape # get the height and width for the alpha channel


    bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] * (1.0 - alpha)).astype('uint8') # emerge with black where its without alpha
    bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] + (fg * alpha)).astype('uint8')  # emerge

    return bg


def cv2_putText_5(img, text, position, fontScale, color, mode=0):

    fontFace = 'DoHyeon-Regular.ttf'
    imgH, imgW = img.shape[:2]
    org = [imgW//(position[0]),imgH//(position[1])]

    # 0 is default, same as cv2.putText()left bottom.　1=left top　2= middle

    # get the text area
    fontPIL = ImageFont.truetype(font = fontFace, size = fontScale)
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (0,0)))
    text_w, text_h = dummy_draw.textsize(text, font=fontPIL)
    text_b = int(0.1 * text_h)

    # get the coordinate of left top of text
    x, y = org
    offset_x = [0, 0, text_w//2]
    offset_y = [text_h, 0, (text_h+text_b)//2]
    x0 = x - offset_x[mode]
    y0 = y - offset_y[mode]
    img_h, img_w = img.shape[:2]

    # if its out of the screen, ignore
    if not ((-text_w < x0 < img_w) and (-text_b-text_h < y0 < img_h)) :
        print ("out of bounds")
        return img

    # get roi of the image that accords to the text area
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0+text_w, img_w), min(y0+text_h+text_b, img_h)

    # put black on the roi(text area) then put the text over it
    text_area = np.full((text_h+text_b,text_w,3), (0,0,0), dtype=np.uint8)
    text_area[y1-y0:y2-y0, x1-x0:x2-x0] = img[y1:y2, x1:x2]

    # make it PIL and make it a text
    imgPIL = Image.fromarray(text_area)
    draw = ImageDraw.Draw(imgPIL)
    draw.text(xy = (0, 0), text = text, fill = color, font = fontPIL)

    # put PIL to OpenCV scale
    text_area = np.array(imgPIL, dtype = np.uint8)

    # the roi(text area) gets replaced by the text
    img[y1:y2, x1:x2] = text_area[y1-y0:y2-y0, x1-x0:x2-x0]

    return img



if __name__ == "__main__":
    main()

