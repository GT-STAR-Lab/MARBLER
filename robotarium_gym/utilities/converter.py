from PIL import Image
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-image_directory', type=str, help='Directory containing the images')
    args = parser.parse_args()

    # Set the directory containing the images
    image_directory = args.image_directory

    # Create a dictionary to store image paths based on episode number
    episodes = {}

    # Iterate over the files in the directory
    for filename in os.listdir(image_directory):
        if filename.endswith(".png"):
            # Extract episode and step numbers from the filename
            episode_num, step_num = filename.split("episode")[1].split("step")[0], filename.split("step")[1].split(".")[0]
            
            # Create episode key if it doesn't exist in the dictionary
            if episode_num not in episodes:
                episodes[episode_num] = []
            
            # Append image path to the corresponding episode key
            episodes[episode_num].append(os.path.join(image_directory, filename))

    # Create GIFs for each episode
    for episode_num, image_paths in episodes.items():
        # Sort image paths based on the step number
        image_paths.sort(key=lambda x: int(x.split("step")[1].split(".")[0]))
        
        # Create a new GIF file for the episode
        gif_path = os.path.join(image_directory, f"episode{episode_num}.gif")
        
        # Open the first image to get its size
        with Image.open(image_paths[0]) as first_image:
            # Create a list to store frames
            frames = []
            
            # Iterate over the image paths and append frames to the list
            for image_path in image_paths:
                with Image.open(image_path) as image:
                    # Resize the image if needed (optional)
                    image = image.resize(first_image.size)
                    
                    # Append the resized image to the frames list
                    frames.append(image.copy())
            
            # Save the frames as a GIF file
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=500, loop=0)
            
            print(f"Created GIF for Episode {episode_num} at {gif_path}")

if __name__ == '__main__':
    main()