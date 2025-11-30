import cv2
import numpy as np
import os

# -------------------------
# CONFIGURATION
# -------------------------
VIDEO_PATH = "x.mp4"  # Change to your video path or use 0 for webcam
CLASSES_FILE = "classes.txt"
MODEL_CFG = "yolov3_testing.cfg"
MODEL_WEIGHTS = "sihax.weights"

# Detection parameters
WHT = 320
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3
KUMANDA_ARALIK = 10

# Drone display name
DRONE_DISPLAY_NAME = "Bayraktar TB2"

# -------------------------
# FILE VERIFICATION
# -------------------------
def verify_files():
    """Check if all required files exist"""
    missing_files = []
    
    if not os.path.exists(VIDEO_PATH) and VIDEO_PATH != 0:
        missing_files.append(f"Video file: {VIDEO_PATH}")
    if not os.path.exists(CLASSES_FILE):
        missing_files.append(f"Classes file: {CLASSES_FILE}")
    if not os.path.exists(MODEL_CFG):
        missing_files.append(f"Model config: {MODEL_CFG}")
    if not os.path.exists(MODEL_WEIGHTS):
        missing_files.append(f"Model weights: {MODEL_WEIGHTS}")
    
    if missing_files:
        print("ERROR: Missing files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure all files are in the correct directory.")
        return False
    return True

# -------------------------
# LOAD CLASS NAMES
# -------------------------
def load_classes():
    """Load class names from file"""
    try:
        with open(CLASSES_FILE, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        print(f"Loaded {len(classes)} classes: {classes}")
        return classes
    except Exception as e:
        print(f"Error loading classes: {e}")
        return []

# -------------------------
# LOAD YOLO MODEL
# -------------------------
def load_model():
    """Load YOLO model"""
    try:
        net = cv2.dnn.readNet(MODEL_WEIGHTS, MODEL_CFG)
        
        # Set backend and target for better performance
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        print("Model loaded successfully!")
        return net
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# -------------------------
# OBJECT DETECTION FUNCTION
# -------------------------
def findObjects(outputs, img, classNames, target_class="uav"):
    """
    Detect objects and return the best drone detection
    """
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    # Collect all detections
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > CONF_THRESHOLD:
                w = int(det[2] * wT)
                h = int(det[3] * hT)
                x = int(det[0] * wT - w / 2)
                y = int(det[1] * hT - h / 2)

                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(bbox, confs, CONF_THRESHOLD, NMS_THRESHOLD)

    # Find the best drone detection
    if len(indices) > 0:
        indices = np.array(indices).flatten()
        
        # Look for drone class specifically
        drone_detection = None
        best_conf = 0
        
        for i in indices:
            class_name = classNames[classIds[i]].lower()
            
            # Check if it's a drone (case-insensitive)
            if target_class.lower() in class_name or "drone" in class_name:
                if confs[i] > best_conf:
                    best_conf = confs[i]
                    drone_detection = i
        
        # If no specific drone found, use the first detection
        if drone_detection is None and len(indices) > 0:
            drone_detection = int(indices[0])
        
        if drone_detection is not None:
            i = drone_detection
            x, y, w, h = bbox[i]

            # Calculate center of bounding box
            cx = x + w // 2
            cy = y + h // 2
            
            # Draw circular detection shape (yellow) - use average of width and height for radius
            radius = max(w, h) // 2 + 10
            cv2.circle(img, (cx, cy), radius, (0, 255, 255), 3)
            
            # Draw crosshair on target
            cross_size = radius // 3
            cv2.line(img, (cx - cross_size, cy), (cx + cross_size, cy), (0, 255, 255), 2)
            cv2.line(img, (cx, cy - cross_size), (cx, cy + cross_size), (0, 255, 255), 2)
            
            # Draw label with custom drone name
            label = f'{DRONE_DISPLAY_NAME} {int(confs[i]*100)}%'
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Position label above the circle
            label_x = cx - text_width // 2
            label_y = cy - radius - 10
            
            # Draw background rectangle for text
            cv2.rectangle(img, (label_x - 5, label_y - text_height - 5), 
                         (label_x + text_width + 5, label_y + 5), (0, 255, 255), -1)
            
            # Draw text
            cv2.putText(img, label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            return x, y, w, h

    return None

# -------------------------
# DRAW UI ELEMENTS (ADAPTIVE)
# -------------------------
def draw_ui(img):
    """Draw targeting UI overlay - adaptive to video orientation"""
    H, W = img.shape[:2]
    
    is_portrait = H > W

    # Outer border (green)
    cv2.rectangle(img, (0, 0), (W-1, H-1), (0, 200, 0), 4)

    if is_portrait:
        # Portrait mode - adjust targeting zone
        # Make it wider relative to height
        margin_x = W // 8  # Smaller horizontal margin
        margin_y_top = H // 6  # More space at top
        margin_y_bottom = H // 6
        
        cv2.rectangle(img, 
                     (margin_x, margin_y_top), 
                     (W - margin_x, H - margin_y_bottom), 
                     (200, 0, 150), 3)
    else:
        # Landscape mode - original layout
        cv2.rectangle(img, (W//4, H//10), (3*W//4, 9*H//10), (200, 0, 150), 3)

    # Center crosshair (green) - scaled to image size
    crosshair_size = min(W, H) // 16
    cv2.line(img, (W//2 - crosshair_size, H//2), (W//2 + crosshair_size, H//2), 
             (0, 255, 0), 2)
    cv2.line(img, (W//2, H//2 - crosshair_size), (W//2, H//2 + crosshair_size), 
             (0, 255, 0), 2)
    
    # Draw center circle
    cv2.circle(img, (W//2, H//2), 3, (0, 255, 0), -1)

# -------------------------
# DRAW TRACKING INFO (ADAPTIVE)
# -------------------------
def draw_tracking_info(img, target):
    """Draw tracking information and guidance - adaptive to orientation"""
    H, W = img.shape[:2]
    x, y, w, h = target
    cx, cy = x + w//2, y + h//2
    
    is_portrait = H > W

    # Draw line from center to target
    cv2.line(img, (W//2, H//2), (cx, cy), (0, 255, 0), 3)

    # Draw target center point
    cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)
    cv2.circle(img, (cx, cy), 12, (0, 255, 0), 2)

    # Calculate horizontal offset
    horiz = cx - W//2
    
    # Calculate vertical offset
    vert = cy - H//2
    
    # Determine text position based on orientation
    if is_portrait:
        text_x = 20
        text_y_start = H//2 - 100
        font_scale = 0.6
    else:
        text_x = W//4 + 10
        text_y_start = H//2 - 80
        font_scale = 1.0
    
    # Display horizontal direction
    if abs(horiz) > 5:  # Deadzone
        if horiz > 0:
            text = f"RIGHT: {horiz/KUMANDA_ARALIK:.2f}"
        else:
            text = f"LEFT: {abs(horiz)/KUMANDA_ARALIK:.2f}"
        cv2.putText(img, text, (text_x, text_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)

    # Display vertical direction
    if abs(vert) > 5:  # Deadzone
        if vert > 0:
            text = f"DOWN: {vert/KUMANDA_ARALIK:.2f}"
        else:
            text = f"UP: {abs(vert)/KUMANDA_ARALIK:.2f}"
        cv2.putText(img, text, (text_x, text_y_start + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)

    # Check if target is inside the targeting zone
    if is_portrait:
        margin_x = W // 8
        margin_y_top = H // 6
        margin_y_bottom = H // 6
        inside = (margin_x < cx < W - margin_x) and (margin_y_top < cy < H - margin_y_bottom)
    else:
        inside = (W//4 < cx < 3*W//4) and (H//10 < cy < 9*H//10)
    
    # Status text position
    status_y = H - 50 if is_portrait else H - 150
    status_font_scale = 1.0 if is_portrait else 1.5
    
    if inside:
        cv2.putText(img, "TARGET LOCKED", (text_x, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, status_font_scale, (0, 255, 0), 3)
    else:
        cv2.putText(img, "ACQUIRING...", (text_x, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, status_font_scale, (0, 0, 255), 3)
    
    # Display distance/size info
    distance_text = f"Size: {w}x{h}"
    cv2.putText(img, distance_text, (text_x, text_y_start + 80),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), 2)

# -------------------------
# MAIN FUNCTION
# -------------------------
def main():
    """Main execution function"""
    
    # Verify all files exist
    if not verify_files():
        return
    
    # Load classes
    classNames = load_classes()
    if not classNames:
        return
    
    # Load model
    net = load_model()
    if net is None:
        return
    
    # Open video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {VIDEO_PATH}")
        print("Try using 0 for webcam or check your video file path")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orientation = "Portrait" if height > width else "Landscape"
    
    print(f"Video opened: {width}x{height} ({orientation}), {fps} FPS, {total_frames} frames")
    
    # Calculate display size to fit screen (max 1280 height for portrait)
    MAX_DISPLAY_HEIGHT = 1000
    MAX_DISPLAY_WIDTH = 1920
    
    if height > width:  # Portrait
        if height > MAX_DISPLAY_HEIGHT:
            scale = MAX_DISPLAY_HEIGHT / height
            display_width = int(width * scale)
            display_height = MAX_DISPLAY_HEIGHT
        else:
            display_width = width
            display_height = height
    else:  # Landscape
        if width > MAX_DISPLAY_WIDTH:
            scale = MAX_DISPLAY_WIDTH / width
            display_height = int(height * scale)
            display_width = MAX_DISPLAY_WIDTH
        else:
            display_width = width
            display_height = height
    
    print(f"Display size: {display_width}x{display_height}")
    
    # Create named window with specific size
    cv2.namedWindow("UAV TRACKING", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("UAV TRACKING", display_width, display_height)
    
    frame_count = 0
    detection_count = 0
    
    # Main processing loop
    while True:
        success, img = cap.read()
        
        if not success:
            print("End of video or cannot read frame")
            break
        
        frame_count += 1
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(img, 1/255, (WHT, WHT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        
        # Get output layer names
        layerNames = net.getLayerNames()
        outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
        
        # Run detection
        outputs = net.forward(outputNames)
        
        # Draw UI overlay
        draw_ui(img)
        
        # Detect and track objects
        target = findObjects(outputs, img, classNames)
        
        if target is not None:
            detection_count += 1
            x, y, w, h = target
            cx, cy = x + w//2, y + h//2
            
            # Print detection info for every frame
            print(f"Frame {frame_count}: DETECTED - Position: ({cx}, {cy}), Size: {w}x{h}, BBox: ({x},{y},{x+w},{y+h})")
            
            draw_tracking_info(img, target)
        else:
            # Show "NO TARGET" message
            H, W = img.shape[:2]
            cv2.putText(img, "NO TARGET", (20, H - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Print no detection for every frame
            print(f"Frame {frame_count}: NO TARGET")
        
        # Show frame counter
        cv2.putText(img, f"Frame: {frame_count}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Resize frame for display if needed
        if img.shape[0] != display_height or img.shape[1] != display_width:
            display_img = cv2.resize(img, (display_width, display_height))
        else:
            display_img = img
        
        # Show frame
        cv2.imshow("UAV TRACKING", display_img)
        
        # Exit on ESC key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('q'):  # Q key
            break
        elif key == ord(' '):  # SPACE - pause
            cv2.waitKey(0)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete:")
    print(f"  Total frames: {frame_count}")
    print(f"  Detections: {detection_count}")
    print(f"  Detection rate: {(detection_count/frame_count*100):.1f}%")

# -------------------------
# RUN PROGRAM
# -------------------------
if __name__ == "__main__":
    main()