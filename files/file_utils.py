import zipfile
import tarfile
import os
import json
import io
import tokenize
from pathlib import Path
from typing import Optional, List

EXCLUDE = {
    'py' : ["__init__.py", "main.py"],
}
def get_extension_files(path: str, extension: str) -> List[str]:
    """
    Retrieves a list of file paths from a given directory that have a specific file extension.

    Args:
        path (str): The directory path to search for files.
        extension (str): The file extension to filter the files by.

    Returns:
        List[str]: A list of file paths that have the specified file extension.
    """
    rootdir = Path(path)
    file_list = sorted([str(f) for f in rootdir.glob(f'**/*.{extension}') if f.is_file() and f not in EXCLUDE[extension]])
    return file_list


def unzip_files(pathes, path_to: Optional[str] = None) -> str:
    """
    Unzips files from the given paths to the specified destination path.

    Args:
       pathes (Union[str, List[str]]): The path(s) of the file(s) to be unzipped.
       path_to (Optional[str]): The destination path where the files will be extracted to.
       If not provided, the current directory will be used.

    Returns:
       str: The path to the extracted files.

    Raises:
       AssertionError: If the file format is not supported (must be .zip or .tar).
    """
    if path_to is None:
        path_to = './data'
    else:
        path_to = Path(path_to)
        path_to.mkdir(exist_ok=True)
    if not isinstance(pathes, list):
        pathes = [pathes]

    for path in pathes:
        print(f"Extracting file: {os.path.basename(path)}...")
        if path.endswith('.tar') or path.endswith('.tar.gz'):
            with tarfile.open(path, 'r') as tar:
                tar.extractall(path_to)
                print(f"File: {os.path.basename(path)} has been successfully unpacked")
        elif path.endswith('.zip'):
            with zipfile.ZipFile(path, 'r') as zip:
                zip.extractall(path_to)
                print(f"File: {os.path.basename(path)} has been successfully unpacked")
        else:
            raise AssertionError("wrong file format. Please use .zip or .tar file")
    return path_to


def remove_comments_and_docstrings(source: str) -> str:
    """
    Removes comments and docstrings from a given source code string.

    Args:
        source (str): The source code string from which to remove comments and docstrings.

    Returns:
        str: The source code string with comments and docstrings removed.

    This function uses the `tokenize` module from the Python standard library to iterate over the tokens in the source code string. It checks each token's type and performs the following actions:
    - If the token is a comment, it is ignored.
    - If the token is a string and it is not preceded by an indent or a newline, and it is not the first token on the line, it is added to the output string.
    - For all other tokens, they are added to the output string.

    The function then joins the lines of the output string and removes any leading or trailing whitespace.

    Note: This function assumes that the source code string is in valid Python syntax.

    Example:
        >>> source = '''
        ... def my_function(arg1, arg2):
        ...     '''This is a docstring.''''
        ...     # This is a comment.
        ...     return arg1 + arg2
        ... '''
        >>> remove_comments_and_docstrings(source)
        'def my_function(arg1, arg2):\\n    return arg1 + arg2\\n'
    """

    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(line for line in out.splitlines() if line.strip())
    return out


def extract_content_from_files(path: str = None,
                               extension: str = 'py',
                               outdir: Optional[bool] = None,
                               remove_docstr: Optional[bool] = True,
                               file_list: Optional[List[str]] = None,
                               ) -> List[str]:
    """
    Extracts content from files in a directory and saves it to a JSON file.

    Args:
        path (str): The path to the directory containing the files.
        extension (str, optional): The file extension to filter the files.
        outdir (Optional[bool], optional): The directory to save result.
        remove_docstr (Optional[bool], optional): Whether to remove docstrings from the files
        file_list (Optional[List[str]], optional): A list of file paths to extract content from.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the content of the files and the list of file paths.

    Raises:
        AssertionError: If there are no files with the specified extension in the directory.
    """
    if file_list is None:
        file_list = get_extension_files(path=path, extension=extension)
        print(f"no .{extension} files in directory {path}") if len(file_list) == 0 else None

    files_content = []

    for file_ in file_list:

        with open(file_, 'r', encoding='utf-8') as f:
            if remove_docstr:
                content = remove_comments_and_docstrings(f.read())
            else:
                content = f.read()

        files_content.append(content)

    if outdir:
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True)
        jsonpath = outdir / 'data.json'
        jsonpath.write_text(json.dumps(files_content))
        print(f"Data has been saved to {jsonpath}")

    return files_content, file_list
