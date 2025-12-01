#!/usr/bin/env python3
"""
ROS Bag Image Rotation Tool

This script processes all bag files in a directory and rotates images from
/camera/color/image_raw topic by 180 degrees.

Usage:
    python bag_head_cam_reverse.py <bag_directory> [--output-dir <output_dir>] [--overwrite]
    
Examples:
    # Process bags in place (overwrite original files)
    python bag_head_cam_reverse.py /home/lab/lerobot_groot/raw_data/groot_train_data/3orb_depalletize_box --overwrite
    
    # Process bags and save to new directory
    python bag_head_cam_reverse.py /home/lab/lerobot_groot/raw_data/groot_train_data/3orb_depalletize_box --output-dir /path/to/output
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Initialize CvBridge for ROS Image <-> OpenCV conversion
bridge = CvBridge()

def rotate_image_180(image_msg):
    """
    Rotate a ROS Image message by 180 degrees.
    
    Args:
        image_msg: sensor_msgs.msg.Image message
        
    Returns:
        sensor_msgs.msg.Image: Rotated image message
    """
    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        
        # Rotate 180 degrees using cv2.rotate (faster than np.rot90)
        # ROTATE_180 rotates the image 180 degrees clockwise
        rotated_cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        
        # Convert back to ROS Image message
        rotated_image_msg = bridge.cv2_to_imgmsg(rotated_cv_image, encoding=image_msg.encoding)
        
        # Preserve header information
        rotated_image_msg.header = image_msg.header
        
        return rotated_image_msg
    except Exception as e:
        print(f"Error rotating image: {e}")
        return image_msg  # Return original if rotation fails

def process_bag_file(input_bag_path, output_bag_path, target_topic='/camera/color/image_raw'):
    """
    Process a single bag file: rotate images from target topic by 180 degrees.
    
    Args:
        input_bag_path: Path to input bag file
        output_bag_path: Path to output bag file
        target_topic: Topic name to process (default: '/camera/color/image_raw')
        
    Returns:
        tuple: (success: bool, image_count: int, error_message: str)
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_bag_path):
            return False, 0, f"Input bag file not found: {input_bag_path}"
        
        # Get bag info
        bag_info = rosbag.Bag(input_bag_path, 'r').get_type_and_topic_info()
        topics = bag_info[1]
        
        # Check if target topic exists
        if target_topic not in topics:
            print(f"⚠️  Warning: Topic '{target_topic}' not found in {os.path.basename(input_bag_path)}")
            print(f"   Available topics: {list(topics.keys())}")
            # Still process the bag, just copy it without modification
            return copy_bag_file(input_bag_path, output_bag_path)
        
        print(f"📦 Processing: {os.path.basename(input_bag_path)}")
        print(f"   Topic '{target_topic}' found with {topics[target_topic].message_count} messages")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_bag_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        image_count = 0
        
        # Process bag file
        with rosbag.Bag(input_bag_path, 'r') as input_bag:
            with rosbag.Bag(output_bag_path, 'w') as output_bag:
                # Process all messages
                for topic, msg, t in input_bag.read_messages():
                    if topic == target_topic:
                        # Rotate image
                        rotated_msg = rotate_image_180(msg)
                        output_bag.write(topic, rotated_msg, t)
                        image_count += 1
                    else:
                        # Copy other messages as-is
                        output_bag.write(topic, msg, t)
        
        print(f"   ✅ Processed {image_count} images from '{target_topic}'")
        return True, image_count, None
        
    except Exception as e:
        error_msg = f"Error processing bag file {input_bag_path}: {str(e)}"
        print(f"   ❌ {error_msg}")
        return False, 0, error_msg

def copy_bag_file(input_bag_path, output_bag_path):
    """
    Copy a bag file without modification (used when target topic doesn't exist).
    
    Args:
        input_bag_path: Path to input bag file
        output_bag_path: Path to output bag file
        
    Returns:
        tuple: (success: bool, image_count: int, error_message: str)
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_bag_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Copy bag file
        with rosbag.Bag(input_bag_path, 'r') as input_bag:
            with rosbag.Bag(output_bag_path, 'w') as output_bag:
                for topic, msg, t in input_bag.read_messages():
                    output_bag.write(topic, msg, t)
        
        print(f"   📋 Copied bag file (no target topic found)")
        return True, 0, None
        
    except Exception as e:
        error_msg = f"Error copying bag file {input_bag_path}: {str(e)}"
        print(f"   ❌ {error_msg}")
        return False, 0, error_msg

def process_directory(bag_directory, output_dir=None, overwrite=False, target_topic='/camera/color/image_raw'):
    """
    Process all bag files in a directory.
    
    Args:
        bag_directory: Directory containing bag files
        output_dir: Output directory (if None and overwrite=False, creates _reversed suffix)
        overwrite: If True, overwrite original files
        target_topic: Topic name to process
        
    Returns:
        dict: Statistics about processing
    """
    bag_directory = Path(bag_directory)
    
    if not bag_directory.exists():
        print(f"❌ Error: Directory not found: {bag_directory}")
        return None
    
    if not bag_directory.is_dir():
        print(f"❌ Error: Path is not a directory: {bag_directory}")
        return None
    
    # Find all bag files
    bag_files = sorted(bag_directory.glob('*.bag'))
    
    if len(bag_files) == 0:
        print(f"⚠️  Warning: No .bag files found in {bag_directory}")
        return None
    
    print(f"\n{'='*80}")
    print(f"🔧 Bag Image Rotation Tool")
    print(f"{'='*80}")
    print(f"📂 Input directory: {bag_directory}")
    print(f"📊 Found {len(bag_files)} bag file(s)")
    print(f"🎯 Target topic: {target_topic}")
    print(f"🔄 Rotation: 180 degrees")
    
    if overwrite:
        print(f"⚠️  Mode: OVERWRITE (original files will be modified)")
        output_dir = bag_directory
    elif output_dir:
        output_dir = Path(output_dir)
        print(f"📁 Output directory: {output_dir}")
    else:
        output_dir = bag_directory.parent / f"{bag_directory.name}_reversed"
        print(f"📁 Output directory: {output_dir} (auto-generated)")
    
    print(f"{'='*80}\n")
    
    # Statistics
    stats = {
        'total_files': len(bag_files),
        'processed': 0,
        'failed': 0,
        'total_images': 0,
        'skipped': 0
    }
    
    # Process each bag file
    for i, bag_file in enumerate(bag_files, 1):
        print(f"[{i}/{len(bag_files)}] Processing: {bag_file.name}")
        
        if overwrite:
            output_bag_path = bag_file
        else:
            output_bag_path = output_dir / bag_file.name
        
        success, image_count, error = process_bag_file(
            str(bag_file),
            str(output_bag_path),
            target_topic=target_topic
        )
        
        if success:
            stats['processed'] += 1
            stats['total_images'] += image_count
            if image_count == 0:
                stats['skipped'] += 1
        else:
            stats['failed'] += 1
            print(f"   Error: {error}")
        
        print()  # Empty line for readability
    
    # Print summary
    print(f"{'='*80}")
    print(f"📊 Processing Summary")
    print(f"{'='*80}")
    print(f"Total files: {stats['total_files']}")
    print(f"✅ Successfully processed: {stats['processed']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"📷 Total images rotated: {stats['total_images']}")
    if stats['skipped'] > 0:
        print(f"⚠️  Skipped (no target topic): {stats['skipped']}")
    print(f"{'='*80}\n")
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description='Rotate images in ROS bag files by 180 degrees',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'bag_directory',
        type=str,
        help='Directory containing bag files to process'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (if not specified and --overwrite not set, creates _reversed suffix directory)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite original bag files (use with caution!)'
    )
    parser.add_argument(
        '--topic',
        type=str,
        default='/camera/color/image_raw',
        help='Target topic to process (default: /camera/color/image_raw)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.overwrite and args.output_dir:
        print("❌ Error: Cannot use both --overwrite and --output-dir")
        sys.exit(1)
    
    # Process directory
    stats = process_directory(
        args.bag_directory,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        target_topic=args.topic
    )
    
    if stats is None:
        sys.exit(1)
    
    if stats['failed'] > 0:
        sys.exit(1)
    
    print("✅ All bag files processed successfully!")

if __name__ == '__main__':
    main()
