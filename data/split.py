def extract(input_file, output_file, bound):
    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for _ in range(bound):
                line = infile.readline()
                if not line:  
                    break
                outfile.write(line)
        print(f"Successfully extracted the required samples from '{input_file}' to '{output_file}'.")

    except FileNotFoundError:
        print(f"The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

input_cv_file = 'dataset.csv'  
output_file = 'smallerData.csv'  
bound = 1500

extract(input_cv_file, output_file, bound)
