import sys 
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import numpy as np
import threading
import time
import random
from random import seed
# define size of image
width, height = 512, 512
halt = False


def displayCallback():
    # Fill image with dummy content
    global image, width, height
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glLoadIdentity()
    
    gl.glRasterPos2i(-1, -1)
    gl.glDrawPixels(width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, image)
    glut.glutSwapBuffers()

def reshapeCallback(width, height): 
    gl.glClearColor(1, 1, 1, 1)
    gl.glViewport(0, 0, width, height)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

def keyboardCallback(key, x, y):     
    if key == b'\033': 
        sys.exit()
        halt = True
    elif key == b'q': 
        sys.exit()
        halt = True



def manipulateImage():  
    milliseconds = int(round(time.time() * 1000))
    print(milliseconds)
    seed(milliseconds)  
    while halt == False:        
        for row in image:
            for item in row:
                num = random.random()
                if num < 0.35:
                    item[1] = 255
                if num > 0.35 and num < 0.74:
                    item[2] = 255
                else:
                    item[3] = 255      
            


if __name__ == "__main__":
        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
        glut.glutInitWindowSize(512, 512)
        glut.glutInitWindowPosition(100, 100)
        glut.glutCreateWindow('Example window')
        glut.glutDisplayFunc(displayCallback)
        glut.glutIdleFunc(displayCallback)
        glut.glutReshapeFunc(reshapeCallback)
        glut.glutKeyboardFunc(keyboardCallback)
        image = np.zeros((width, height, 4), dtype=np.ubyte)

        manip_thread = threading.Thread(target=manipulateImage)
        manip_thread.start()


# Create image
glut.glutMainLoop()
manip_thread.join()
