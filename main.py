import cv2
import numpy as np
import matplotlib.pyplot as plt
import inspect
import argparse
import sys

MIN_MATCH_COUNT = 3

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_video", help="path to the (optional) video file")
args = vars(ap.parse_args())


def main():
    if not args.get("input_video", False):
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(args["input_video"])

    video = cv2.VideoCapture('C:\\Users\\szymo\\Desktop\\MWR\\MWRpila\\VID_20200330_140318.mp4')

    if not video.isOpened():
        print('Error opening video')
        sys.exit()

    # query image
    saw_image = cv2.imread('pila3F.jpg')

    show_img(saw_image)
    saw_image_in_gray = cv2.cvtColor(saw_image, cv2.COLOR_BGR2GRAY)


    orb = cv2.ORB_create()
    fast = cv2.FastFeatureDetector_create()
    fast.setNonmaxSuppression(False)
    saw_image_kp, saw_image_des = orb.detectAndCompute(saw_image_in_gray, None)

    saw_kp_vis = cv2.drawKeypoints(saw_image_in_gray, saw_image_kp, outImage=None, color=(0, 255, 0),
                                   flags=None)
    show_img(saw_kp_vis)

    while video.isOpened():
        # Capture frame-by-frame
        # frame - target image
        _, frame = video.read()

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)

        
        frame_in_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # At this moment i have characteristic points and a descriptor for every one of them of saw image and
        # characteristic points and a descriptor for every one of them of frames from given video
        # Now I need to match keypoints from image and frames to detect saw on video

        frame_kp, frame_des = orb.detectAndCompute(frame_in_gray, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(saw_image_des, frame_des)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m in matches:
            if m.distance < 35:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([saw_image_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w, d = saw_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            print("matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        matches_vis = cv2.drawMatches(saw_image, saw_image_kp, frame, frame_kp, good, None, **draw_params)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 1920, 1080)
        cv2.imshow('frame', matches_vis)

    video.release()
    cv2.destroyAllWindows()


def show_img(img, bw=False):
    fig = plt.figure(figsize=(13, 13))
    ax = fig.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(img, cmap='Greys_r' if bw else None)  # color map settings
    plt.show()


def print_interesting_members(obj):
    for name, value in inspect.getmembers(obj):
        try:  # changing values to float type
            float(value)  # we want values that can be represented as numerical values
            print(f'{name} -> {value}')
        except Exception:  # if it is not possible we pass it
            pass


if __name__ == '__main__':
    main()
