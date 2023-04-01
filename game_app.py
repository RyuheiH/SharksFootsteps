import cv2
import time
import glob
import numpy as np
import pygame.mixer
from PIL import Image, ImageDraw, ImageFont




def main():
    pygame.mixer.init() #initial
    while True:
        pygame.mixer.music.load("music/氷雨.mp3") #bgm
        enter_sound = pygame.mixer.Sound("music/決定ボタンを押す8.mp3")  # サウンドをロード
        pygame.mixer.music.play(1) #-1 means infinite loop
        key = startpage()
        if key == 13: #enter key
            enter_sound.play()  # サウンドを再生
            camera()

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

        if ret == False: #when the shark video is done
            cap.set(cv2.CAP_PROP_POS_FRAMES,0) #it sets back the video
            ret, img = cap.read() #and reads it again

        cv2.imshow(videoname, img) #show img
        key = cv2.waitKey(1) & 0xFF #waits for the key command

        if key == ord('q'): #when q is pressed, it quits the while loop
            cv2.destroyAllWindows()
            break
        elif key == 13:
            cv2.destroyAllWindows()



def show_screenshot():
    files = glob.glob("screenshot/capture_*.png")
    n = 0

    for n in files:
        img = cv2.imread(n)

        imgH, imgW = img.shape[:2]
        position = [imgW//2,(imgH//1)-(100)]
        fontPIL = 'DoHyeon-Regular.ttf'
        text = "Press Enter to see other pictures"

        img = cv2_putText_5(img = img,text = text,org = position,fontFace = fontPIL,fontScale = 30,color = (255, 255, 255),mode = 2)
        
        cv2.imshow(f'{n}',img)
        
        key = cv2.waitKey(0) & 0xFF #waits for the key command

        if key == ord('q'): #when q is pressed, it quits the while loop
            cv2.destroyAllWindows()
            break
        elif key == 13:
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



def merge_images(bg, fg_alpha, s_x, s_y): #2つの画像のサイズが一致してないとダメ
    alpha = fg_alpha[:,:,3]  # アルファチャンネルだけ抜き出す(要は2値のマスク画像)
    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) # grayをBGRに
    alpha = alpha / 255.0    # 0.0〜1.0の値に変換

    fg = fg_alpha[:,:,:3]

    f_h, f_w, _ = fg.shape # アルファ画像の高さと幅を取得
    #b_h, b_w, _ = bg.shape # 背景画像の高さを幅を取得

    # 画像の大きさと開始座標を表示
    #print("f_w:{} f_h:{} b_w:{} b_h:{} s({}, {})".format(f_w, f_h, b_w, b_h, s_x, s_y))

    bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] * (1.0 - alpha)).astype('uint8') # アルファ以外の部分を黒で合成
    bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] + (fg * alpha)).astype('uint8')  # 合成

    return bg




def camera():
        cap = cv2.VideoCapture(0)#("IMG_1908.MOV") #video from camera
        capShark = cv2.VideoCapture("videos/shark.mp4") #video of Shark
        imgDH = cv2.imread('graphic/imgDH_180.png',-1)
        smile_count = 0
        smilewait = []
        face_detector = cv2.FaceDetectorYN.create("recognition_trained_data/yunet.onnx", "", (960,510))
        cascade_smile = cv2.CascadeClassifier("recognition_trained_data/haarcascade_smile.xml") #data for smile
        screenshot = False
        smileframe = -40

        pygame.mixer.music.load("music/水中流星群.mp3") #bgm 
        pygame.mixer.music.play(-1) #-1 means infinite loop
        pygame.mixer.music.set_volume(0.3)

        while True:
            ret, frame = cap.read()
            retS, imgS = capShark.read()
            img = cv2.resize(frame,(960,510)) 
            img = cv2.flip(img, 1) #mirror the camera to make it like a mirror

            frame_num = capShark.get(cv2.CAP_PROP_POS_FRAMES)
            #print(frame_num)

            if retS == False: #when the shark video is done
                capShark.set(cv2.CAP_PROP_POS_FRAMES,0) #it sets back the video
                retS, imgS = capShark.read() #and reads it again


            faces = facial_recognition(face_detector,img)

            if len(faces) > 0:
                if frame_num > 280 and frame_num < 380:
                    play_the_shark_attack()
                    break
                elif frame_num > 1020 and frame_num < 1150:  
                    play_the_shark_attack()
                    break
                
                imgS, smile_count, smilewait, screenshot = multiple_faces(faces,img,imgS,cascade_smile,imgDH,smile_count, smilewait)
                    
            if screenshot == True:
                smileframe = capShark.get(cv2.CAP_PROP_POS_FRAMES)
                #print("Screenshot true")

            if (capShark.get(cv2.CAP_PROP_POS_FRAMES)) == smileframe or (capShark.get(cv2.CAP_PROP_POS_FRAMES)) < smileframe + 30:
                imgH, imgW = imgS.shape[:2]
                position = [imgW//2,imgH//2]
                fontPIL = 'DoHyeon-Regular.ttf'
                text = "Took Screenshot!"
                imgS = cv2_putText_5(img = imgS,text = text,org = position,fontFace = fontPIL,fontScale = 60,color = (255, 255, 255),mode = 2)

                   
            cv2.imshow('Shark',imgS)

            key = cv2.waitKey(1) & 0xFF #waits for the key command
            if key == ord('q'): #when q is pressed, it quits the while loop
                break

        cap.release()
        cv2.destroyAllWindows()




def play_the_shark_attack():
    capAttack = cv2.VideoCapture("videos/shark_attack.mp4")
    #capAttack.set(cv2.CAP_PROP_FPS, 30)
    
    pygame.mixer.music.load("music/se_bukimi2-1.mp3") #bgm 
    pygame.mixer.music.play(0) #-1 means infinite loop
    pygame.mixer.music.set_volume(0.2)

    #fps = capAttack.get(cv2.CAP_PROP_FPS)
    #print(fps)

    while True:

        retA, imgA = capAttack.read()
        
        if retA == False: #when the shark video is done
                break
        
        imgH, imgW = imgA.shape[:2]
        position = [imgW//2,imgH//4]
        fontPIL = 'DoHyeon-Regular.ttf'
        size = 60
        text = "Found By Shark!"
        color = (85, 84, 234) #RGBではなくBGR234, 84, 85)

        imgA = cv2_putText_5(img = imgA,text = text,org = position,fontFace = fontPIL,fontScale = size,color = color,mode = 2)
        
        if retA == False: 
            break
        
        cv2.imshow('GameOver',imgA)
        cv2.waitKey(30)

    capAttack.release()
    cv2.destroyAllWindows()
     



def multiple_faces(faces,img,imgS,cascade_smile,imgDH,smile_count, smilewait):
    i = 0
    screenshot = False

    for face in faces:
        #print(i)
        #print(face)
        trim = trim_the_detected_face(face,img)
        imgS = put_displays_all_together(imgDH,imgS,trim,i)
        smiles = smile_recognition(img,cascade_smile,face)
        #print(smiles)
        
        smiles = smiles if smiles is not None else []
        if len(smiles) > 0 :
            smile_count, smilewait, screenshot = smile_screenshot(smiles,face,smile_count,smilewait,imgS)

        i = i + 1
        
    return imgS, smile_count, smilewait, screenshot




def trim_the_detected_face(face,img):
            
            x = int(face[0])
            y = int(face[1])
            w = int(face[2])
            h = int(face[3])

            trim = img[y:y+h,x:x+w]   # range of x and y of face rectangle 
            
            if np.any(trim) :
                trim = cv2.resize(trim,(100,100))  #resize the window for the face
                
            return trim



def smile_screenshot(smiles,face,smile_count,smilewait,imgS):

    #print(smile_count)
    #print(smilewait)
    screenshot = False

    if smile_count == 0: #when its the first smile, it just takes the screenshot
        smilewait.append(time.time())
        smile_count += 1
        cv2.imwrite(f"screenshot/capture_{smile_count}.png", imgS) #when smile is detected, takes screenshot
        #print("smiled")
        screenshot = True

    if time.time() - smilewait[smile_count-1] > 3 and smile_count > 0: #when it is not first time of screenshot, it cannot take screenshot until 3 seconds passes since the last screenshot
        smilewait.append(time.time())
        smile_count += 1
        cv2.imwrite(f"screenshot/capture_{smile_count}.png", imgS) #when smile is detected, takes screenshot
        #print("smiled")
        screenshot = True

    """
    for(sx,sy,sw,sh) in smiles:
        x = face[0]
        y = face[1]
        #cv2.circle(imgS,(int(x+sx+sw/2),int(y+sy+sh/2)),int(sw/2),(0, 0, 255),2)#red
    """

    return smile_count, smilewait, screenshot



def facial_recognition(face_detector,img):
        
    _, faces = face_detector.detect(img)
    faces = faces if faces is not None else []

    return faces



def smile_recognition(img,cascade_smile,face):
            
            x = int(face[0])
            y = int(face[1])
            w = int(face[2])
            h = int(face[3])

            roi = img[y:y+h, x:x+w] #for the smile recognition
            if np.any(roi):
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #gray scale video for recognition
                smiles = cascade_smile.detectMultiScale(roi_gray,scaleFactor= 1.2, minNeighbors=30, minSize=(100, 100))#笑顔識別

                return smiles




def put_displays_all_together(imgDH,imgS,trim,i):

    # 処理領域を設定(left(x1), top(y1), right(x2), bottom(y2))
    roi_for_DH = [[0,20,180,200],[180,20,360,200],[360,20,540,200],[540,20,720,200],[720,20,900,200],[900,20,1080,200]]
    x_offset = [40,220,400,580]
    y_offset = [40,40,40,40]

    h, w = trim.shape[:2]
    cut = 3/2
    trim_cropped = trim[0:round(h/cut), 0:round(w), :]

    imgS[y_offset[i]:y_offset[i]+trim_cropped.shape[0], x_offset[i]:x_offset[i]+trim_cropped.shape[1]] = trim_cropped

    s_roi = imgS[roi_for_DH[i][1]: roi_for_DH[i][3], roi_for_DH[i][0]: roi_for_DH[i][2]]
    s_roi = merge_images(s_roi,imgDH,0,0)
    imgS[roi_for_DH[i][1]: roi_for_DH[i][3], roi_for_DH[i][0]: roi_for_DH[i][2]] = s_roi

    return imgS
     




def cv2_putText_5(img, text, org, fontFace, fontScale, color, mode=0):
# cv2.putText()にないオリジナル引数「mode」　orgで指定した座標の基準
# 0（デフォ）＝cv2.putText()と同じく左下　1＝左上　2＝中央

    # テキスト描写域を取得
    fontPIL = ImageFont.truetype(font = fontFace, size = fontScale)
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (0,0)))
    text_w, text_h = dummy_draw.textsize(text, font=fontPIL)
    text_b = int(0.1 * text_h) # バグにより下にはみ出る分の対策

    # テキスト描写域の左上座標を取得（元画像の左上を原点とする）
    x, y = org
    offset_x = [0, 0, text_w//2]
    offset_y = [text_h, 0, (text_h+text_b)//2]
    x0 = x - offset_x[mode]
    y0 = y - offset_y[mode]
    img_h, img_w = img.shape[:2]

    # 画面外なら何もしない
    if not ((-text_w < x0 < img_w) and (-text_b-text_h < y0 < img_h)) :
        print ("out of bounds")
        return img

    # テキスト描写域の中で元画像がある領域の左上と右下（元画像の左上を原点とする）
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0+text_w, img_w), min(y0+text_h+text_b, img_h)

    # テキスト描写域と同サイズの黒画像を作り、それの全部もしくは一部に元画像を貼る
    text_area = np.full((text_h+text_b,text_w,3), (0,0,0), dtype=np.uint8)
    text_area[y1-y0:y2-y0, x1-x0:x2-x0] = img[y1:y2, x1:x2]

    # それをPIL化し、フォントを指定してテキストを描写する（色変換なし）
    imgPIL = Image.fromarray(text_area)
    draw = ImageDraw.Draw(imgPIL)
    draw.text(xy = (0, 0), text = text, fill = color, font = fontPIL)

    # PIL画像をOpenCV画像に戻す（色変換なし）
    text_area = np.array(imgPIL, dtype = np.uint8)

    # 元画像の該当エリアを、文字が描写されたものに更新する
    img[y1:y2, x1:x2] = text_area[y1-y0:y2-y0, x1-x0:x2-x0]

    return img



if __name__ == "__main__":
    main()

