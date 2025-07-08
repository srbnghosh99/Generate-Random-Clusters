import bson
import os
import time

MAX_BSON_SIZE_MB = 1500 * 1024 * 1024  # 100 MB per file (adjustable)
MAX_BSON_SIZE = 1500 * 1024 * 1024

def create_folder(outdirectory):
    print('create folder')
    try:
        os.mkdir(outdirectory)
        print(f"Directory '{outdirectory}' created successfully")
    except FileExistsError:
        print(f"Directory '{outdirectory}' already exists")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_solutions_v1(filename):
    """Load list of solutions from a BSON file."""
    with open(filename, "rb") as f:
        data = bson.decode(f.read())
    return data["solutions"]

def size_of_bson(filename):
    """Load list of solutions from a BSON file."""
    with open(filename, "rb") as f:
        data = bson.decode(f.read())
    # print(data)

    solutions = [item["solution"] for item in data["data"]]
    
    return len(solutions)

def load_solutions(filename):
    """Load list of solutions from a BSON file."""
    with open(filename, "rb") as f:
        data = bson.decode(f.read())
    # print(data)

    solutions = [item["solution"] for item in data["data"]]
    fitness_values = [item["fitness"] for item in data["data"]]
    return solutions, fitness_values

def write_to_file(outfile, text):
    with open(outfile, "a") as file:
        file.write(text + "\n")

def save_bson_chunks(data_list, outdir,function,outputfile):
    start = time.time()
    chunk = []
    total_size = 0
    file_index = 0
    create_folder(outdir)
    for entry in data_list:
        # Estimate the BSON size of this entry
        temp_bson = bson.BSON.encode({"data": [entry]})
        entry_size = len(temp_bson)

        # If the chunk would exceed the limit, save and reset
        base_filename = outdir + '/' + 'selected_solutions_' + function
        if total_size + entry_size > MAX_BSON_SIZE and chunk:
            with open(f"{base_filename}_{file_index}.bson", "wb") as f:
                f.write(bson.BSON.encode({"data": chunk}))
            print(f"✅ Saved chunk {file_index} with {len(chunk)} entries.")
            write_to_file(outputfile,f'✅ Saved chunk {file_index} with {len(chunk)} entries.')
            chunk = []
            total_size = 0
            file_index += 1

        # Add entry to current chunk
        chunk.append(entry)
        total_size += entry_size

    # Save the final chunk
    if chunk:
        with open(f"{base_filename}_{file_index}.bson", "wb") as f:
            f.write(bson.BSON.encode({"data": chunk}))
        print(f"✅ Saved final chunk {file_index} with {len(chunk)} entries.")
    end = time.time()
    elapsed_time = end - start
    elapsed_time = elapsed_time / 60
    # print(f"Total Elapsed time: {elapsed_time:.6f} minutes")
    write_to_file(outputfile, f'The Selection process took: {elapsed_time:.6f} minutes')

# def save_solutions(outdir,combined):
#     print(outfile)
#     """Save list of solutions to a BSON file."""
#
#     outfile = os.path.splitext(outfile)[0]
#     filename = outfile + '.bson'
#     print(filename)
#     with open(filename, "wb") as f:
#         f.write(bson.BSON.encode({"data": combined}))
        # f.write(bson.encode({"solutions": master_populations}))



