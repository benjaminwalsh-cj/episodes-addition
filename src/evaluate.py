import os
import logging
import typing
import joblib
import numpy as np
import pandas as pd

logger = logging.basicConfig(level='INFO',
                             format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                             datefmt='%H:%M:%S')


def load_evaluation_scores(path: str, label: str) -> pd.DataFrame:
    '''Load a precision_recall_fscore_file.xlsx from the path
    Args:
        path (`str`): path to evaluation scores file
        label (`str`): label to distinguish the pipeline run
    Returns:
        Evaluation results dataframe formatted for further evaluation steps
    Raises:
        FileNotFoundError
    '''

    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        logger.error('Path to evaluation file is incorrect.')
        raise

    df['label'] = label

    return_df = df.melt(id_vars=['label', 'Category'], value_vars=[
                        'Precision', 'Recall', 'fscore'], value_name='values', var_name='measure')

    return_df['values'] = return_df['values'].astype(float)

    return return_df


def eval_evaluation_scores(
        pre_scores: pd.DataFrame,
        post_scores: pd.DataFrame,
        pre_label: str,
        post_label: str) -> pd.DataFrame:
    '''Create an evaluation dataframe to easily compare the evaluation results from
    different pipeline runs
    Args:
        pre_scores (`pd.DataFrame`): testing evaluation results from base run
        post_scores (`pd.DataFrame`): testing evaluation results from new run
        pre_label (`str`): value of the `label` column in the pre dataframe
        post_label (`str`): value of the `label` column in the post dataframe
    Return:
        Pandas dataframe for easier cross-runs comparisons of eval results
    '''

    if not isinstance(pre_scores, pd.DataFrame):
        logger.error('Argument `pre_scores` is not a pandas dataframe.')
        raise TypeError

    if not isinstance(post_scores, pd.DataFrame):
        logger.error('Argument `post_scores` is not a pandas dataframe.')
        raise TypeError

    # Merge on category and measure
    merged_df = pre_scores.merge(post_scores, on=['Category', 'measure'])

    # Drop label columns
    merged_df = merged_df[['Category', 'measure', 'values_x', 'values_y']]

    # Add delta
    merged_df['delta'] = merged_df['values_y'] - merged_df['values_x']

    # Flag if improved
    merged_df['improve_flag'] = np.where(merged_df['delta'] > 0, 1, 0)

    # Relabel columns
    merged_df = merged_df.rename(
        columns={
            'values_x': f'{pre_label}_value',
            'values_y': f'{post_label}_value'
        }
    )

    merged_df = merged_df.sort_values(by=['Category', 'measure'])

    return merged_df


def load_confusion_matrix(path: str) -> pd.DataFrame:
    '''Load a persisted confusion matrix

    Args:
        path (`str`): path to confusion matrix
    Returns:
        Pandas dataframe of the confusion matrix
    '''
    cm = pd.read_excel(path)
    cm = cm.rename(
        columns={
            'Unnamed: 0': 'Category'
        }
    )
    return cm


def eval_confusion_matrices(
        pre_matrix: pd.DataFrame,
        post_matrix: pd.DataFrame,
        pre_label: str,
        post_label: str,
        col_count: int) -> pd.DataFrame:
    '''Generate an evaluation of the confusion matrices produced between pipeline runs

    Args:
        pre_matrix (`pd.DataFrame`): matrix from original run
        post_matrix (`pd.DataFrame`): matrix from changed run
        pre_label (`str`): label for original run
        post_label (`str`): label from changed run
    Returns:
        Evaluation dataframe
    '''
    pre_matrix_copy = pre_matrix.copy(deep=True)
    post_matrix_copy = post_matrix.copy(deep=True)

    # Add labels to category values
    pre_matrix_copy['Category'] = pre_label + '_' + pre_matrix_copy['Category']
    post_matrix_copy['Category'] = post_label + \
        '_' + pre_matrix_copy['Category']

    # Add empty row to separate confusion matrices for greater readability
    # row = pd.DataFrame([['', '' , '']] , columns=pre_matrix_copy.columns)
    empty_array = np.empty(shape=(1, col_count))
    empty_array[:] = np.nan
    row = pd.DataFrame(empty_array, columns=pre_matrix_copy.columns)
    # row = row.fillna('')

    # Concat matrices together
    return_df = pd.concat([pre_matrix_copy, row])
    return_df = pd.concat([return_df, post_matrix_copy])

    return return_df


def load_predictions(path: str) -> pd.DataFrame:
    '''Load a prediction file
    Args:
        path (`str`): path to the prediction file
    Returns:
        Dataframe of the predictions
    '''

    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        logger.error('Could not find predictions at specified path.')
        raise

    return df


def gen_class_counts(pred_df: pd.DataFrame, label: str) -> pd.DataFrame:
    '''Count the number of predictions for each class in the passed df
    Args:
        pred_df (`pd.DataFrame`): prediction dataframe
        label (`str`): label of the predictions' run
    Returns:
        Pandas dataframe of counts
    '''

    # Generate counts
    counts = pred_df['label_1'].value_counts()

    # Store counts as a dataframe
    counts_df = pd.DataFrame(counts)

    # Add label
    counts_df['label'] = label

    # Extract category from index
    counts_df['category'] = counts_df.index

    # Relabel
    counts_df = counts_df.rename(
        columns={
            'label_1': 'counts'
        }
    )

    # Reset index and reduce columns
    counts_df = counts_df.reset_index()[
        ['label', 'category', 'counts']
    ]

    return counts_df


def eval_class_counts(
        pre_counts: pd.DataFrame,
        post_counts: pd.DataFrame,
        pre_label: str,
        post_label: str) -> pd.DataFrame:
    '''Generate evaluation dataframe for class counts across runs
    Args:
        pre_counts (`pd.DataFrame`): class counts in initial pipeline run
        post_counts (`pd.DataFrame`): class counts in changed pipeline run
        pre_label (`str`): string label of initial run
        post_label (`str`): string label of changed run
    Returns:
        Evaluation dataframe
    '''

    # Merge dfs
    merge_df = pre_counts.merge(post_counts, on='category', how='inner')

    # Add delta
    merge_df['delta'] = merge_df['counts_y'] - merge_df['counts_x']

    # Relabel
    merge_df = merge_df.rename(
        columns={
            'counts_x': f'{pre_label}_count',
            'counts_y': f'{post_label}_count'
        }
    )

    # Select columns
    merge_df = merge_df[
        [
            'category',
            f'{pre_label}_count',
            f'{post_label}_count',
            'delta'
        ]
    ]

    return merge_df


def compare_labels(
        pre_preds: pd.DataFrame,
        post_preds: pd.DataFrame,
        pre_label: str,
        post_label: str,
        changed_label_flag: str) -> pd.DataFrame:
    '''Identify and isolate providers who changed labels between pipeline
    runs
    Args:
        pre_preds (`pd.DataFrame`): initial pipeline run predictions
        post_preds (`pd.DataFrame`): changed pipeline run predictions
        pre_label (`str`): string label of initial run
        post_label (`str`): string label of changed run
        changed_label_flag (`str`) ('y', 'n'): whether to produce a dataframe
        of providers who's label changed or remained constant across runs
    Returns:
        Dataframe containing providers who changed and their respective labels and
        probabitlies
    '''

    # Merge together
    merge_df = pre_preds.merge(post_preds, how='inner', on='npi')

    if changed_label_flag == 'y':
        filtered_merge_df = merge_df[merge_df['label_1_x']
                                     != merge_df['label_1_y']]
    else:
        filtered_merge_df = merge_df[merge_df['label_1_x']
                                     == merge_df['label_1_y']]

    return_df = filtered_merge_df[
        [
            'npi',
            'label_1_x',
            'probability_1_x',
            'label_1_y',
            'probability_1_y'
        ]
    ].rename(
        columns={
            'label_1_x': f'{pre_label}_label_1',
            'probability_1_x': f'{pre_label}_probability_1',
            'label_1_y': f'{post_label}_label_1',
            'probability_1_y': f'{post_label}_probability_1'
        }
    )

    return return_df


def gen_switched_label_subset(changed_labels_df: pd.DataFrame, train_data_path: str = None) -> pd.DataFrame:
    '''Generate a randomized selection of providers who changed labels for manual evaluation
    Args:
        changed_labels_df (`pd.Dataframe`): dataframe of providers who changed labels between runs
        train_data_path (`str`): path to the joblib file storing the training dataframe. This can be used
        to get the ground truth for a selected NPI if it was available in the training set
    Returns:
        Dataframe of randomized selection of providers who changed labels
    '''

    npi_selection = list(np.random.choice(
        changed_labels_df['npi'], size=30, replace=False))

    return_df = changed_labels_df[changed_labels_df['npi'].isin(npi_selection)]

    if train_data_path is None:
        return_df['manual_label'] = ''

    # Get the ground truth if it was available in the training subset
    else:
        train_data = joblib.load(train_data_path)
        train_data = train_data[train_data['npi'].isin(npi_selection)][[
            'npi', 'label']]
        return_df = return_df.merge(train_data, how='left', on='npi')
        return_df = return_df.rename(
            columns={
                'label': 'manual_label'
            }
        )
    return return_df


def gen_proba_stats(
        df: pd.DataFrame,
        proba_col: str,
        category_col: str = None,
        category_value: str = None) -> pd.Series:
    '''Evaluate the distribution of the model's probabilities for label 1 in
    a provided dataframe.

    Args:
        df (`pd.DataFrame`): dataframe to evaluate. Should have class and probabilities cols
        proba_col (`str`): string name of the probability column
        category_col (`str`): string name of the category column
        category_value (`str`): category to to subset to before generating stats
    Returns:
        Series description of the probability columns
    '''

    if category_col is None:
        summary = df[proba_col].describe()
    else:
        try:
            filtered_df = df[df[category_col] == category_value]
        except KeyError:
            logger.error(
                'Provided category value is not in provided category column.')

        summary = filtered_df[proba_col].describe()

    return summary


def eval_proba_distributions(
        pre_series: pd.Series,
        post_series: pd.Series,
        pre_label: str,
        post_label: str) -> pd.DataFrame:
    '''Generate an evaluation dataframe comparing the probability stats across runs
    Args:
        pre_series (`pd.Series`): Summary stats of label 1 probabilities in initial run
        post_series (`pd.Series`): Summary stats of label 1 probabilities in changed run
        pre_label (`str`): Label for initial run
        post_label (`str`): Label for changed run
    Returns:
        Dataframe of summary stats
    '''

    return_df = pd.DataFrame()

    # Add initial run's summary stats
    return_df[pre_label] = pre_series

    # Add changed run's summary stats
    return_df[post_label] = post_series

    # Calculate deltas
    return_df['delta'] = return_df[post_label] - return_df[pre_label]

    return return_df


def gen_evaluation_report(
        comparison_df_list: list[pd.DataFrame],
        comparison_df_labels: list[str],
        output_dir: str,
        file_name: str) -> None:
    '''Write overall report based on the provided comparison dfs. Persists
    dfs to output directory as well.

    Args:
        comparison_df_list (`list[pd.DataFrame]`): list of evaluation dataframes to include in report
        comparison_df_labels (`list[str]`): list of labels for evaluation dataframes
        output_dir (`str`): output directory for report and persisted dataframes
        file_name (`str`): file name for evaluation report
    Returns:
        None
    '''

    # Generate output dir if doesn't exist
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

    if os.path.isdir(f'{output_dir}persisted_dataframes/') == False:
        os.mkdir(f'{output_dir}persisted_dataframes/')

    if len(comparison_df_labels) != len(comparison_df_list):
        logger.error(
            'List of dataframes must be the same length as list of dfs.')
        raise ValueError

    full_path = output_dir + file_name + '.txt'

    with open(full_path, 'w') as file:

        for df, label in zip(comparison_df_list, comparison_df_labels):
            file.write(label)
            file.write('\n')
            file.write(df.to_string())
            file.write('\n\n')
    file.close()

    for df, label in zip(comparison_df_list, comparison_df_labels):
        output_path = f'{output_dir}persisted_dataframes/{label.replace(" ", "_")}.xlsx'
        df.to_excel(output_path)
