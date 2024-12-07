import cv2


cap = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:

    ret, frame = cap.read()
    
    if not ret:
        break
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
 
    gray = cv2.equalizeHist(gray)  
    gray = cv2.GaussianBlur(gray, (5, 5), 0) 
    

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    

    for (x, y, w, h) in faces:
   
        center = (x + w // 2, y + h // 2)
        radius = max(w, h) // 2
        cv2.circle(frame, center, radius, (255, 0, 0), 2)
        cv2.putText(frame, 'Rosto', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    

    cv2.imshow('Frame', frame)
    
    # Sair do loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
