# -*- coding: utf-8 -*-
"""
Simple icon generator for ArcheoGlyph.
Run this script to create a placeholder icon.
"""

from PIL import Image, ImageDraw

def create_icon():
    """Create a simple placeholder icon."""
    size = 64
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a stylized jar/amphora shape
    # Main body
    draw.ellipse([16, 24, 48, 56], fill='#CD7F32', outline='#8B4513', width=2)
    
    # Neck
    draw.rectangle([26, 14, 38, 26], fill='#CD7F32', outline='#8B4513', width=1)
    
    # Rim
    draw.ellipse([22, 10, 42, 20], fill='#DEB887', outline='#8B4513', width=1)
    
    # Sparkle to indicate "glyph" magic
    draw.polygon([(52, 8), (56, 16), (52, 24), (48, 16)], fill='#FFD700')
    draw.polygon([(48, 12), (56, 16), (48, 20), (44, 16)], fill='#FFD700')
    
    return img

if __name__ == '__main__':
    icon = create_icon()
    icon.save('icon.png')
    print("Icon created: icon.png")
