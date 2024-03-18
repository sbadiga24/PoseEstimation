import cv2 as cv
import numpy as np
import os
import keyboard

def main():
    # Start video capture
    cap = cv.VideoCapture(2)
          
    # VIDEO MODE	OUTPUT RESOLUTION (SIDE BY SIDE)	FRAME RATE (FPS)	FIELD OF VIEW
    # HD2K	        4416x1242	                            15	                Wide
    # HD1080	    3840x1080	                            30, 15	            Wide
    # HD720	        2560x720	                            60, 30, 15	        Extra Wide
    # VGA	        1344x376	                            100, 60, 30, 15 	Extra Wide
    #Resolution set to HD1080 
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Split the frame into left and right images
        left_right_image = np.split(frame, 2, axis=1)

        # Display images
        cv.imshow("left_Frame", left_right_image[0])
        cv.imshow("right_Frame", left_right_image[1])

        key = cv.waitKey(1) & 0xFF

        if keyboard.is_pressed('q'):
            print("Quitting...")  # Debugging: Confirm 'q' was pressed
            break


    # Cleanup
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()