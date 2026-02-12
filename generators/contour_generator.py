# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Contour Generator (Auto Trace)
Extracts artifact silhouette from input image using OpenCV Canny Edge Detection.
"""

from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import Qt
try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

class ContourGenerator:
    """
    Generates SVG contours from images using OpenCV Canny Edge Detection.
    """
    def __init__(self):
        pass

    def generate(self, image_path, style=None, color=None, symmetry=False):
        """
        Generates a contour SVG from the image.
        
        :param image_path: Path to the input image
        :param style: (Unused in Auto Trace, kept for interface compatibility)
        :param color: Hex color code (e.g. "#FF0000"). If None, extracts dominant color.
        :return: SVG string
        """
        if cv2 is None or np is None:
            raise ImportError(
                "OpenCV and NumPy are required for Auto Trace. "
                "Please install them via 'pip install opencv-python-headless numpy' "
                "or use the OSGeo4W Shell."
            )

        # Read image using OpenCV
        stream = open(image_path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError("Failed to load image.")

        # 1. Preprocessing & Resizing
        # We process at a reasonable size for speed, but SVG should map back to logic size.
        original_h, original_w = img.shape[:2]
        max_dim = 1500 # Increased for better detail
        scale_factor = 1.0
        
        processing_img = img
        if max(original_h, original_w) > max_dim:
            scale_factor = max_dim / max(original_h, original_w)
            new_w = int(original_w * scale_factor)
            new_h = int(original_h * scale_factor)
            processing_img = cv2.resize(img, (new_w, new_h))
        
        # 2. Shadow Removal (HSV Strategy)
        # Shadows are usually Low Saturation + High Value (Light Grey).
        # Artifacts are High Saturation OR Low Value (Dark lines/Dark objects).
        
        if len(processing_img.shape) == 4:
            # Drop alpha for HSV calc, assume white bg
            bgr = cv2.cvtColor(processing_img, cv2.COLOR_BGRA2BGR)
        else:
            bgr = processing_img
            
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Mask 1: Significant Color (Saturation > 20)
        # Most artifacts (pottery, bronze) have some color. Shadows are grey (S near 0).
        s_mask = cv2.threshold(s, 20, 255, cv2.THRESH_BINARY)[1]
        
        # Mask 2: Dark Dark Lines (Value < 90)
        # If it's black/dark brown, keep it regardless of saturation.
        v_mask = cv2.threshold(v, 90, 255, cv2.THRESH_BINARY_INV)[1]
        
        # Combine: Keep if Colored OR Dark
        target_mask = cv2.bitwise_or(s_mask, v_mask)
        
        # Clean up mask (Open/Close)
        kernel = np.ones((3,3), np.uint8)
        target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel, iterations=4) # Close small holes
        
        # 3. Find Contours
        contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return '<svg></svg>'
            
        # Extract Dominant Color (from MASKED area)
        final_color = color
        if final_color is None:
            # Use the mask to get only object pixels
            final_color = self._extract_dominant_color(processing_img, target_mask)

        # Largest contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Simplify
        epsilon = 0.0015 * cv2.arcLength(main_contour, True) 
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # 4. Generate SVG
        # Style Logic
        is_line_drawing = style and ("Line" in style or "선화" in style)
        
        svg_w = processing_img.shape[1]
        svg_h = processing_img.shape[0]
        
        svg_output = []
        svg_output.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}">')
        
        path_data = ""
        if len(approx) > 2:
            # SYMMETRY LOGIC
            # If symmetry is requested, we mirror the LEFT side to the RIGHT.
            # 1. Find the centerline (x_center) of the bounding rect
            x, y, w_rect, h_rect = cv2.boundingRect(approx)
            center_x = x + w_rect / 2
            
            final_points = []
            
            if symmetry:
                # Filter points to keep only those on the LEFT of center_x
                # Actually, simple mirroring might be better:
                # 1. Take all points. 
                # 2. Reflect them across center_x.
                # 3. Valid approach for artifacts: Input assumption is "Left side is good" or "Whole shape is roughly centered".
                # Let's assume we want to enforce bilateral symmetry based on the detected shape's center.
                
                # Better approach: 
                # Process the contour points. For every point (x, y), create a point (2*center_x - x, y).
                # But we need a continuous path.
                # Let's split the contour at Top and Bottom intersections with centerline.
                # This is complex geometry.
                
                # Simpler robust approach for "Artifact Illustration":
                # Average the left and right sides? No.
                # Just take the LEFT half and mirror it to the RIGHT.
                
                # 1. Center the contour at 0,0
                points = approx.reshape(-1, 2)
                
                # Find top and bottom points to define axis
                top_pt = min(points, key=lambda p: p[1])
                bottom_pt = max(points, key=lambda p: p[1])
                
                # Vertical axis x
                axis_x = (top_pt[0] + bottom_pt[0]) / 2
                
                # Gather left side points (x <= axis_x)
                left_side = [p for p in points if p[0] <= axis_x]
                
                # Sort them by Y to ensure drawing order (Top to Bottom)
                # This might break complex shapes like handles, but for pots/arrowheads it's usually fine.
                # Let's stick to original order but filter? No, order matters.
                
                # Re-ordering strategy:
                # 1. Find top-most point index.
                # 2. Walk through contour. If x < axis, keep it. If x > axis, ignore.
                # 3. Generate mirrored points for the kept points.
                
                # Let's try a simpler visual trick:
                # Mirror the IMAGE first? No, we already have contours.
                # Mirror the contour points.
                
                left_contour = []
                for pt in points:
                    if pt[0] < axis_x:
                        left_contour.append(pt)
                
                if len(left_contour) < 2:
                     # Fallback if shape is weird
                     final_points = points.tolist()
                else:
                    # Sort by Y (Top -> Bottom) creates a "half-profile"
                    left_sorted = sorted(left_contour, key=lambda p: p[1])
                    
                    # Create Right side (Reflect X)
                    right_side = []
                    for pt in reversed(left_sorted): # Bottom -> Top
                        reflected_x = int(axis_x + (axis_x - pt[0]))
                        right_side.append([reflected_x, pt[1]])
                        
                    final_points = left_sorted + right_side
                    # Close loop
                    final_points.append(left_sorted[0])
            else:
                final_points = approx.reshape(-1, 2).tolist()

            # Build Path Data
            if len(final_points) > 2:
                start = final_points[0]
                path_data += f"M {start[0]},{start[1]} "
                for i in range(1, len(final_points)):
                    pt = final_points[i]
                    path_data += f"L {pt[0]},{pt[1]} "
                path_data += "Z"
            
        if is_line_drawing:
            # Line Drawing: No Fill, Black Stroke
            svg_output.append(
                f'<path d="{path_data}" '
                f'fill="none" '
                f'stroke="#000000" '
                f'stroke-width="3" '
                f'stroke-linecap="round" '
                f'stroke-linejoin="round"/>'
            )
        else:
            # Silhouette: Filled Color, Optional Stroke
            svg_output.append(
                f'<path d="{path_data}" '
                f'fill="{final_color}" '
                f'fill-opacity="1.0" '
                f'stroke="{final_color}" ' # Stroke same as fill for smoothness
                f'stroke-width="1" '
                f'stroke-linecap="round" '
                f'stroke-linejoin="round"/>'
            )
            
        svg_output.append("</svg>")
        
        return "".join(svg_output)

    def _extract_dominant_color(self, img, mask=None):
        """
        Extracts dominant color, strictly filtering for 'colorful' pixels first.
        """
        try:
            if len(img.shape) == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
            # Convert to HSV to filter
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Filter Criteria:
            # 1. Mask must be active (if provided)
            # 2. Saturation > 15 (Avoid grey/shadows)
            # 3. Value > 40 (Avoid pitch black) and Value < 230 (Avoid white highlights)
            
            valid_mask = np.ones_like(h, dtype=bool)
            if mask is not None:
                valid_mask = (mask > 0)
                
            # Refined filter for Celadon/Pottery
            # We want the 'body' color, not the highlight (white) or shadow (black).
            color_mask = (s > 15) & (v > 40) & (v < 240) & valid_mask
            
            # Get pixels
            pixels = img[color_mask]
            
            # Fallback: If strict filter removes everything (e.g. very grey artifact), relax it
            if len(pixels) < 50:
                pixels = img[valid_mask] # Just use the shape mask
                
            if len(pixels) < 10:
                return "#8B4513" # Fallback if empty
                
            pixels = np.float32(pixels)
            
            # Top 3 Colors to avoid outlier
            n_colors = 3
            if len(pixels) < n_colors: n_colors = 1
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Pick the most saturated center? Or just the most abundant?
            # K-Means returns centers. Labels tells abundance.
            # Let's pick the center with highest Saturation to avoid muddy colors.
            
            best_color = centers[0]
            max_sat = -1
            
            for center in centers:
                # Convert BGR center to HSV to check saturation
                c_uint8 = np.uint8([[center]])
                c_hsv = cv2.cvtColor(c_uint8, cv2.COLOR_BGR2HSV)[0][0]
                if c_hsv[1] > max_sat:
                    max_sat = c_hsv[1]
                    best_color = center
            
            dom_color = best_color.astype(int)
            return "#{:02x}{:02x}{:02x}".format(dom_color[2], dom_color[1], dom_color[0])
            
        except Exception as e:
            # print(f"Color error: {e}")
            return "#8B4513"
