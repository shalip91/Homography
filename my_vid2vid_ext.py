import os
import cv2


def video_to_image_seq(vid_path, output_path='./datasets/OTB/img/Custom/'):
    os.makedirs(output_path, exist_ok=True)
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0
    print("converting video to frames...")
    while success:
        fname = str(count).zfill(4)
        cv2.imwrite((f"{output_path}raw_frame{count}.jpg"), image)
        success, image = vidcap.read()
        count += 1
    print("total frames: ", count)
    return count

def image_seq_to_video(frames_number, imgs_path, output_path='./video.mp4', fps=24.0):
    output = output_path
    img_array = []
    for i in range(frames_number):
        img = cv2.imread(f"{imgs_path}modified_frame{i}.jpg")
        height, width, layers = img.shape
        img = cv2.resize(img, (width, height))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    print(size)
    print("writing video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, fps, size)
    # out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("saved video @ ", output)

def faceRecognition(frame, scaleFactor, minNeighbors, minSize, maxSize):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    detected_image = frame.copy()
    gray = cv2.cvtColor(detected_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=scaleFactor,
                                          minNeighbors=minNeighbors,
                                          minSize=minSize, maxSize=maxSize)
    font = cv2.FONT_HERSHEY_COMPLEX

    for (x, y, h, w) in faces:
        cv2.rectangle(detected_image, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
        if (x<detected_image.shape[1]//2):
            detected_image = cv2.putText(detected_image, 'Guy bahalal',
                                         (x, y-10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            detected_image = cv2.putText(detected_image, 'haizar',
                                         (x, y - 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return detected_image

if __name__ == '__main__':
    print('my_vid2vid_ext')


"""original video into frames"""
frames_number = video_to_image_seq('video/guy_in_space.mp4', output_path='video/raw_frames/')

"""frame faces detection"""
for i in range(frames_number):
    frame = cv2.imread(f'video/raw_frames/raw_frame{i}.jpg')
    detected = faceRecognition(frame, scaleFactor=1.2,
                                      minNeighbors=4,
                                      minSize=(40,40), maxSize=(70,70))
    cv2.imwrite(f'video/modified_frames/modified_frame{i}.jpg', detected)

"""modified frames into video"""
image_seq_to_video(153, 'video/modified_frames/', output_path='video/guy_in_space_modified.mp4', fps=24.0)
#
