import sys 
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import numpy as np
# define size of image
width, height = 512, 512


def displayCallback():
    # Fill image with dummy content
    global image, width, height
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glLoadIdentity()

    for i in range(0, width):
        image[i][i][0] = 255
        image[width - i - 1][i][1] = 255  
        # Draw image
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
    elif key == b'q': 
        sys.exit()

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


# Create image
image = np.zeros((width, height, 4), dtype=np.ubyte)
glut.glutMainLoop()
