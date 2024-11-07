import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480

imgaesave = False
model = load_model('C:/Users/amrut/University of Groningen - Document/My spam work on code/MNIST dataset/bestmodel.h5')

white = (255,255,255)
black = (0, 0, 0)
red = (255, 0, 0)

labels = {
    0:'Zero',
    1:'One',
    2:'Two',
    3:'Three',
    4:'Four',
    5:'Five',
    6:'Six',
    7:'Seven',
    8:'Eight',
    9:'Nine'
}

iswriting = False
num_xcord = []
num_ycord = []

boudry = 5
predit =True
img_cnt =1
#First we initialse the pygame model
pygame.init()
#Setting the window size
display_surface = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
#Setting the font for the window
font = pygame.font.Font(None, 18)
pygame.display.set_caption("Digit Board")

#To retain the window 
#This is the part of the code where we write so we can write
#on the screen and recognise the digit
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit() 
    
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(display_surface, white, (xcord, ycord), 4, 0)   
            num_xcord.append(xcord)
            num_ycord.append(ycord) 
        
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
        
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            num_xcord = sorted(num_xcord)
            num_ycord = sorted(num_ycord)

            rect_min_x, rect_max_x = max(num_xcord[0] - boudry, 0), min(WINDOWSIZEX, num_xcord[-1] + boudry)
            rect_min_y, rect_max_y = max(num_ycord[0] - boudry, 0), min(WINDOWSIZEY, num_ycord[-1] + boudry)
            
            num_xcord =[]
            num_ycord =[]

            img_array = np.array(pygame.PixelArray(display_surface))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
            #to save the image
            if imgaesave:
                cv2.imwrite('image.png')
                img_cnt +=1

            if predit:
                image = cv2.resize(img_array, (28,28))
                image = np.pad(image, (10,10), 'constant', constant_values=0)
                image = cv2.resize(image, (28,28))/255

                label = str(labels[np.argmax(model.predict(image.reshape(1,28,28,1)))])

                textSurface = font.render(label, True, red, white)
                textRectObj = textSurface.get_rect()
                textRectObj.left, textRectObj.bottom = rect_min_x, rect_max_y

                display_surface.blit(textSurface, textRectObj)
            
            if event.type == KEYDOWN:
                if event.unicode == 'n':
                    display_surface.fill(black)

        pygame.display.update()




