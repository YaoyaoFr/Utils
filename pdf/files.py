import os


def get_file_paths_within_directory(directory_path: str,
                                    post_fix: str = None):
    # File path
    file_path_list = []

    files = os.listdir(directory_path)
    for file_name in files:
        current_path = os.path.join(directory_path, file_name)
        if os.path.isdir(current_path):
            file_path_list.extend(get_file_paths_within_directory(directory_path=current_path, post_fix=post_fix))
        else:
            if post_fix is not None:
                name, extension = os.path.splitext(file_name)
                if extension != post_fix:
                    continue

            file_path_list.append(current_path)

    return file_path_list
