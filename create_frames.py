import os
import cv2

def process_video(input_path, output_path, max_frames=200, target_resolution=(160, 90)):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 1280 and height == 720:
        os.makedirs(output_path, exist_ok=True)
        frame_count = 0
        success, frame = cap.read()

        while success and frame_count < max_frames:
            frame_count += 1
            resized_frame = cv2.resize(frame, target_resolution)
            frame_filename = os.path.join(output_path, f"frame{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, resized_frame)
            success, frame = cap.read()

        cap.release()

if __name__ == "__main__":
    input_directory = 'dataset/videos'
    output_directory = 'dataset/frames'

    for filename in os.listdir(input_directory):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_directory, filename)
            video_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_directory, video_name)
            process_video(input_path, output_path)

    print("Processing complete.")