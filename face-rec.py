#This Python script performs face recognition in a video using FaceNet and MTCNN. 
#It compares all detected faces in the video with a reference face image and marks identified faces in the output video.
#Only one image of the reference face is used for recognition

import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from tqdm import tqdm


#SET HERE FILE NAMES AND PATHS
image_path = 'reference_face_1.jpg'  # Reference face image
video_path = 'input_video.mp4'  # Input video file
output_video_path = 'output_video.mp4'  # Output video file

#SET HERE SIMILARITY
similarity = 0.6 #similarity threshold

############################################################

# Determine the device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load face detection (MTCNN) and face recognition (ResNet) models
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def get_embedding(img):
    """
    Extracts the face embedding using the ResNet model.
    :param img: PIL Image of a face
    :return: Tensor embedding of the face, or None if detection fails
    """
    try:
        img_cropped = mtcnn(img)
        if img_cropped is None:
            return None

        if img_cropped.ndimension() == 4:  # Multiple faces detected
            embeddings = [resnet(face.unsqueeze(0).to(torch.float32)).squeeze(0) for face in img_cropped]
            return embeddings  # Return a list of embeddings

        return resnet(img_cropped.unsqueeze(0).to(torch.float32)).squeeze(0)  # Single face embedding

    except RuntimeError as e:
        print(f"[Error in get_embedding] {e}")
        return None


def process_video():
    """
    Main function for processing the video and identifying faces.
    """
    # Load reference face image and get its embedding
    face_img = Image.open(image_path)
    face_embedding = get_embedding(face_img)
    if face_embedding is None:
        raise ValueError("Failed to obtain an embedding for the reference face image.")

    # Open video file for processing
    cap = cv2.VideoCapture(video_path)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Setup video writer for output
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL Image format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        boxes, _ = mtcnn.detect(frame_pil)  # Detect faces

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face_crop = frame_pil.crop((x1, y1, x2, y2))
                face_embedding_current = get_embedding(face_crop)

                if face_embedding_current is not None:
                    # Ensure reference embedding is a single tensor
                    if isinstance(face_embedding, list):
                        face_embedding = face_embedding[0]

                    # Compare detected face embedding with reference
                    if isinstance(face_embedding_current, list):  # Multiple faces in the frame
                        for emb in face_embedding_current:
                            emb = emb.unsqueeze(0)
                            distance = torch.nn.functional.cosine_similarity(face_embedding.unsqueeze(0), emb)
                            color, label = ((0, 255, 0), 'Identified') if distance.item() > similarity else ((0, 0, 255), 'Not identified')
                    else:  # Single face in the frame
                        distance = torch.nn.functional.cosine_similarity(face_embedding.unsqueeze(0), face_embedding_current.unsqueeze(0))
                        color, label = ((0, 255, 0), 'Identified') if distance.item() > similarity else ((0, 0, 255), 'Not identified')

                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write processed frame to output video
        out.write(frame)
        pbar.update(1)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pbar.close()


if __name__ == '__main__':
    process_video()
