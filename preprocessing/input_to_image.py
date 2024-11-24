import cv2, os

# Buat fungsi agar bisa dipanggil di codebase main (nantinya)
def get_frames_by_input_video(
    pathInputVideo, pathOutputImage="dataset/video_to_images", framePerSecond=None
):
    if not os.path.exists(pathInputVideo):
        print(f"Path file {pathInputVideo} tidak valid")
        return

    # Buat folder jika belum ada
    os.makedirs(pathOutputImage, exist_ok=True)
    
    # Hapus file lama di folder output
    for filename in os.listdir(pathOutputImage):
        filepath = os.path.join(pathOutputImage, filename)
        os.remove(filepath)
    
    # Buka video
    vidcap = cv2.VideoCapture(pathInputVideo)
    
    # Ambil frame rate dari video
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    if framePerSecond is None:
        framePerSecond = video_fps  # Jika framePerSecond tidak ditentukan, ambil dari video
    print(f"Frame rate dari video: {video_fps} FPS, menyimpan setiap {framePerSecond} frame per detik")
    
    # Hitung interval frame (jika ingin setiap frame, set framePerSecond ke video_fps)
    interval = int(video_fps // framePerSecond) if framePerSecond else 1
    
    count = 0
    saved_frame_count = 1

    while True:
        success, image = vidcap.read()
        if not success:
            break

        # Simpan frame setiap interval
        if count % interval == 0:
            cv2.imwrite(f"{pathOutputImage}/frame{saved_frame_count}.jpg", image)
            saved_frame_count += 1
        
        count += 1

    vidcap.release()
    # print(f"Berhasil menyimpan {saved_frame_count - 1} frame di {pathOutputImage}")