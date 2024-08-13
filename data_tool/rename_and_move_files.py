import os
import shutil

def rename_and_move_files(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize a counter
    counter = 1

    # Iterate through the files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is a .jpg file
        if filename.endswith('.jpg'):
            # Get the base name without extension
            base_name = os.path.splitext(filename)[0]
            jpg_file = os.path.join(input_folder, base_name + '.jpg')
            xml_file = os.path.join(input_folder, base_name + '.txt')

            # Check if the corresponding .xml file exists
            if os.path.exists(xml_file):
                # Define new file names
                new_jpg_name = f"{counter}.jpg"
                new_xml_name = f"{counter}.txt"

                # Define paths for new files
                new_jpg_path = os.path.join(output_folder, new_jpg_name)
                new_xml_path = os.path.join(output_folder, new_xml_name)

                # Move and rename the files
                shutil.move(jpg_file, new_jpg_path)
                shutil.move(xml_file, new_xml_path)

                # Increment the counter
                counter += 1
            else:
                print(f"Corresponding txt file for {jpg_file} not found.")

if __name__ == "__main__":
    rename_and_move_files('./old', './new')
    print("Script completed.")
