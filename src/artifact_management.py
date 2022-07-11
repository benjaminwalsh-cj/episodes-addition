'''This module provides functionality to persist artifacts generated in the pipeline in a systematic and maintainable manner
'''

import os
import shutil
import logging
import pandas as pd
import numpy as np
import joblib

logger = logging.basicConfig(level='INFO',
                             format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                             datefmt='%H:%M:%S')


def generate_index_label(path: str, run_type: str) -> str:
    '''This function generates the index label for artifacts
    Args:
        path (`str`): destination path for the artifacts (run_type-level directory)
        run_type (`str`): the episodes level the pipeline was executed with
    Returns:
        String label to use in indexing new artifacts
    '''

    if os.path.isdir(path) == False:
        logger.error(
            "Provded path does not exist and cannot be used in indexing.")
        raise ValueError

    existing_indexes = []
    for directory in os.listdir(path):
        # Don't count anything that isn't a directory (e.g. .DS_Store)
        if os.path.isdir(path + directory) == True:
            existing_indexes.append(int(directory))

    # Add one to the max index already in use
    new_index = max(existing_indexes) + 1

    # Define the new index label for the directory and the artifacts
    if new_index <= 9:
        new_label = f'{run_type}_0{new_index}'
    else:
        new_label = f'{run_type}_{str(new_index)}'

    return new_label


def generate_artifact_directories(path: str, index_label: str) -> None:
    '''Generate the directories required for pipeline artifact persistence
    Args:
        path (`str`): path to the run_level-type directory
        index_label (`str`): label for the directories indicating the index (run) of the pipeline
    Returns:
        None
    '''

    # Establish index root directory
    index_path = path + index_label
    try:
        os.mkdir(index_path)
    except FileExistsError:
        logger.error('Indexed directory already exists')

    logger.info('Directory successfully created.')


def generate_artifact_dest_path(artifact_name: str, dest: str, index_label: str) -> str:
    '''Create the destination path for a provided artifact using its corresponding index label
    Args:
        artifact_name (`str`): name of the artifact (e.g. its existing file name)
        dest (`str`): destination directory of the artifact
        index_label (`str`): index of the pipeline's run. Should be generated using the
        `generate_index_label` function.
    Returns:
        String full path for the artifact as a shutil.copy dest value
    '''

    dest_path = f'{dest}{artifact_name}_{index_label}'

    return dest_path


def persist_artifact(src: str, dest: str) -> None:
    '''Copy a single artifact to the persistence directory
    Args:
        src (`str`): Source path to the artifact
        dest (`str`): Destination path for the copied artifact
    Returns:
        None
    '''

    shutil.copy(src=src, dst=dest)


def persist_artifact_directory(src: str, dst: str, index_label: str) -> None:
    '''Persist an entire existing directory of artifacts
    Args:
        src (`str`): path to the existing artifact directory
        dst (`str`): path to the desired output artifact directory
        index_label (`str`): index label for the pipeline run
    Returns:
        None
    '''

    # Copy the entire tree
    try:
        shutil.copytree(src, dst)
    except FileExistsError:
        # logger.error('Cannot copy directory tree. Directories already exist in destination.')
        raise

    # Iterate through the copied files and relabel them to use the indexing scheme
    for root, dirs, files in os.walk(dst):
        for name in files:
            if name != '.DS_Store':
                name_parts = name.split('.')
                new_name = f'{name_parts[0]}_{index_label}.{name_parts[1]}'
                old_path = os.path.join(root, name)
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
