import numpy as np
import cv2
import torch

class ObjectDetector:
    def __init__(self):
        # Camera and object parameters
        self.focal = 665  # Focal length in pixels
        self.width = 7.1  # Real-world width of the object in cm
        
        # OpenCV text parameters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.6
        self.color = (0, 0, 255)
        self.thickness = 2
        
        # Initialize YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.eval()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)  # Use 0 for default camera
        if not self.cap.isOpened():
            raise ValueError("Failed to open camera")
            
        # Set up display window
        cv2.namedWindow('Object Distance Measurement', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Distance Measurement', 700, 600)

    def get_dist(self, rectangle_params):
        """Calculate distance from camera based on object width in pixels"""
        pixels = rectangle_params[1][0]
        if pixels > 0:
            dist = (self.width * self.focal) / pixels
        else:
            dist = float('inf')
        return dist

    def get_3d_coordinates(self, x, y, dist):
        """Calculate 3D coordinates from 2D image coordinates and distance"""
        Z = dist
        X = (x - 320) * Z / self.focal
        Y = (y - 240) * Z / self.focal
        return X, Y, Z

    def annotate_image(self, image, bbox, dist, coords_3d):
        """Draw bounding box and information on the image"""
        x1, y1, x2, y2 = bbox
        X, Y, Z = coords_3d
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw labels
        label = f'Coords: ({x1}, {y1}), ({x2}, {y2})'
        distance_label = f'Distance: {dist:.2f} cm'
        coords_3d_label = f'3D Coords: (X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}) cm'
        
        # Position text
        cv2.putText(image, label, (x1, y1 - 40), self.font, self.fontScale, 
                    self.color, self.thickness, cv2.LINE_AA)
        cv2.putText(image, distance_label, (x1, y1 - 20), self.font, self.fontScale, 
                    self.color, self.thickness, cv2.LINE_AA)
        cv2.putText(image, coords_3d_label, (x1, y1), self.font, self.fontScale, 
                    self.color, self.thickness, cv2.LINE_AA)
        return image

    def process_frame(self):
        """Process a single frame from the camera"""
        ret, img = self.cap.read()
        if not ret:
            return False, None

        # Perform object detection
        results = self.model(img)
        detections = results.xyxy[0].cpu().numpy()

        # Process each detection
        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            bbox = (x1, y1, x2, y2)
            bbox_width = x2 - x1

            # Calculate distance and 3D coordinates
            dist = self.get_dist(((0, 0), (bbox_width, 0), 0))
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            coords_3d = self.get_3d_coordinates(center_x, center_y, dist)

            # Annotate image
            img = self.annotate_image(img, bbox, dist, coords_3d)

            # Print information to console
            print(f'Object 2D Coordinates: ({x1}, {y1}), ({x2}, {y2})')
            print(f'Distance from Camera: {dist:.2f} cm')
            print(f'3D Coordinates: (X={coords_3d[0]:.2f}, Y={coords_3d[1]:.2f}, Z={coords_3d[2]:.2f}) cm')

        return True, img

    def run(self):
        """Main loop for continuous camera processing"""
        try:
            while True:
                success, frame = self.process_frame()
                if not success:
                    print("Failed to grab frame")
                    break

                cv2.imshow('Object Distance Measurement', frame)

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = ObjectDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")