from multiprocessing import dummy
import os
import logging
import typing
import joblib
from matplotlib.pyplot import draw_if_interactive
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level='INFO',
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
        logging.error('Path to evaluation file is incorrect.')
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
        logging.error('Argument `pre_scores` is not a pandas dataframe.')
        raise TypeError

    if not isinstance(post_scores, pd.DataFrame):
        logging.error('Argument `post_scores` is not a pandas dataframe.')
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
        logging.error('Could not find predictions at specified path.')
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


def gen_switched_label_subset(
        changed_labels_df: pd.DataFrame,
        train_data_path: str = None) -> pd.DataFrame:
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
            logging.error(
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


def gen_summary_stats(
        dummy_df: pd.DataFrame,
        id_vars: list[str],
        var_name: str) -> pd.DataFrame:
    '''Generate the summary statitics for a pivoted dataframe.
    This function was built to evaluate a dummy episodes dataframe,
    but it can be used with any other dummy pivoted dataframe.
    Args:
        dummy_df (`pd.DataFrame`): pd.dummy dataframe with values to evaluate
        id_vars (`list[str]`): list of col names to be the id variables in the
        melt
        var_name (`str`): name of the unpivoted variable col
    Returns:
        An unpivoted dataframe of summary statistics that can be joined
        to a similar dataframe from a different pipeline run.
    '''

    summary_stats = dummy_df.describe()

    # `describe()` stores the measure in the index but it should be a col for
    # the melt
    summary_stats['measure'] = summary_stats.index

    # Add measure order to maintain order down the line if desired
    summary_stats['measure_order'] = range(8)  # there are 8 stats

    # melt
    return_df = summary_stats.melt(
        id_vars=id_vars,
        var_name=var_name
    )

    # Ensure `value` is a float
    return_df['value'] = return_df['value'].astype(float)

    return return_df


def eval_summary_stats(
        summary_stats_pre: pd.DataFrame,
        summary_stats_post: pd.DataFrame,
        pre_label: str,
        post_label: str) -> pd.DataFrame:
    '''Generate a dataframe comparing the summary stats for
    two different runs of many columns. Originally built to compare
    stats from episodes queries.
    Args:
        summary_stats_pre (`pd.DataFrame`): summary stats from initial run
        summary_stats_post (`pd.DataFrame`): summary stats from changed run
        pre_label (`str`): Label for initial run
        post_label (`str`): Label for changed run
    Returns:
        Dataframe comparing the two statistics summaries
    '''
    # Outer join the two dfs in case episodes appear in one and not the other
    return_df = pd.merge(
        summary_stats_pre,
        summary_stats_post,
        on=['measure', 'episode', 'measure_order'],
        how='outer'
    )

    # Add delta
    return_df['delta'] = return_df['value_y'] - return_df['value_x']

    # Relabel
    return_df = return_df.rename(
        columns={
            'value_x': f'{pre_label}_value',
            'value_y': f'{post_label}_value'
        }
    )

    # Reorder columns
    return_df = return_df[
        [
            'measure_order',
            'episode',
            'measure',
            f'{pre_label}_value',
            f'{post_label}_value',
            'delta'
        ]
    ]

    return return_df


def gen_merged_stats_df(
        eval_df: pd.DataFrame,
        stats_df_list: typing.List[pd.DataFrame],
        stats_df_labels: typing.List[str]) -> pd.DataFrame:
    '''Merge a stats evaluation dataframe with secondary stats dfs
    Args:
        eval_df (`pd.DataFrame`): evaluation dataframe
        stats_df_list (`list[pd.DataFrame]`): list of secondary stats dfs
        stats_df_labels (`list[str]`): list of labels for secondary dfs
    Returns:
        Merged dataframe
    '''

    if len(stats_df_list) != len(stats_df_labels):
        logging.error('`stats_df_list` must be same length as `stats_df_labels'
                      )
        raise ValueError

    merge_df = eval_df.merge(
        stats_df_list[0],
        on=['episode', 'measure'],
        how='left'
    ).rename(
        columns={
            'value': f'{stats_df_labels[0]}_value'
        }
    ).drop(
        'measure_order',
        axis=1
    )

    for df_idx in range(1, len(stats_df_list)):
        merge_df = merge_df.merge(
            stats_df_list[df_idx],
            on=['episode', 'measure'],
            how='left'
        ).rename(
            columns={
                'value': f'{stats_df_labels[df_idx]}_value'
            }
        ).drop(
            'measure_order',
            axis=1
        )

    return merge_df


def gen_mutual_cols_list(
        df_1: pd.DataFrame,
        df_2: pd.DataFrame,
        cols_to_check: typing.Optional[typing.List[str]] = None) -> typing.List[str]:
    '''Find a list of mututal columns
    Args:
        df_1 (`pd.DataFrame`): first dataframe to check
        df_2 (`pd.DataFrame`): second dataframe to check
        cols_to_check (`List[str]`): manual list of columns to check. If not
        `None`, returned list of mutual columns will be max(len(cols_to_check))
    Returns:
        List of string column names of cols that are in both dataframes
    '''
    # find columns that both dfs have (unable to compare otherwise)
    if cols_to_check is None:
        cols = [col for col in df_1.columns if col in df_2.columns]
    # if manual list to check is passed, check that each col in present in dfs
    else:
        cols = []
        for col in cols_to_check:
            try:
                df_1[col]
                df_2[col]
            except KeyError:
                logging.error('Column %s is not in one of the passed dataframes.',
                              col)
                continue
            cols.append(col)

    return cols


def gen_distribution_results(
        df_1: pd.DataFrame,
        df_2: pd.DataFrame,
        cols: typing.List[str]) -> typing.Tuple[list, list]:
    '''Generate the ks t-test statistics and their resepective p-values for the
    provided cols by comparing their distributions in the two provided 
    dataframes.
    Args:
        df_1 (`pd.DataFrame`): first dataframe to evaluate
        df_2 (`pd.DataFrame`): second dataframe to evaluate
        cols (`list[str]`): list of cols to compare
    Returns:
        Tuple of lists containing the results for each of the columns
    '''

    ks_stats = []
    ks_p_values = []
    t_stats = []
    t_p_values = []
    # Generate ks statistic and p-value for each column
    for col in cols:
        # subset to only the column
        df_1_col = df_1[col]
        df_2_col = df_2[col]

        # calculate the ks results
        try:
            ks_results = stats.kstest(df_1_col, df_2_col)
            t_results = stats.ttest_ind(df_1_col, df_2_col)
        except ValueError:
            logging.error('Could not generate results for column %s. It may not'
                          ' be continuous.', col)
            continue

        # add results to respective storage arrays
        ks_stats.append(ks_results[0])
        ks_p_values.append(ks_results[1])
        t_stats.append(t_results[0])
        t_p_values.append(t_results[1])

    return ks_stats, ks_p_values, t_stats, t_p_values


def eval_distribution_tests(
        dummy_df_1: pd.DataFrame,
        dummy_df_2: pd.DataFrame,
        label_1: str,
        label_2: str,
        cols_to_check: typing.Optional[typing.List[str]] = None) -> pd.DataFrame:
    '''Perform a two-sided ks and t tests on episodes cols from two different
    dummy
    dataframes
    Args:
        dummy_df_1 (`pd.DataFrame`): first dummy dataframe with episodes to
        check
        dummy_df_2 (`pd.DataFrame`): second dummy dataframe with episodes to
        check
        label_1 (`str`): label for dummy_df_1
        label_2 (`str`): label for dummy_df_2
        col_to_check (optional: `list[str]`): list of columns to compare.
        Default is `None`, so all continuous columns will be used.
    '''
    # deep copy passed dataframes
    df_1 = dummy_df_1.copy(deep=True)
    df_2 = dummy_df_2.copy(deep=True)

    cols = gen_mutual_cols_list(
        df_1=df_1,
        df_2=df_2,
        cols_to_check=cols_to_check
    )

    # Instantiate empty return dataframe and empty arrays to store values
    return_df = pd.DataFrame()

    ks_stats, ks_p_value, t_stat, t_p_value = gen_distribution_results(
        df_1=df_1,
        df_2=df_2,
        cols=cols
    )

    # Format dataframe
    return_df['episode'] = cols
    return_df['df_1'] = label_1
    return_df['df_2'] = label_2
    return_df['ks_value'] = ks_stats
    return_df['ks_p_value'] = ks_p_value
    return_df['t_statistic'] = t_stat
    return_df['t_p_value'] = t_p_value

    # Filter to only statistically significant values
    return_df = return_df[
        (return_df['ks_p_value'] <= .05) |
        (return_df['t_p_value'] <= .05)
    ]

    # Sort on KS value
    return_df = return_df.sort_values(by='ks_value', ascending=False)

    if return_df.empty == True:
        if len(cols) == 0:
            empty_description = 'No mutual columns'
        elif len(list(filter(lambda x: x <= .05, ks_p_value))) == 0:
            empty_description = 'No significantly different distributions'
        else:
            empty_description = 'Unclear why dataframe is empty'
        return_df = pd.DataFrame(
            {
                'episode': empty_description,
                'df_1': np.nan,
                'df_2': np.nan,
                'ks_value': np.nan,
                'ks_p_value': np.nan,
                't_statistic': np.nan,
                't_p_value': np.nan
            },
            index=[0]
        )

    return return_df


def eval_missing_npis(
        subspecialties: typing.List[str],
        pre_label: str,
        post_label: str,
        missing_npi_counts_array: typing.List[typing.List[int]],
        eval_counts_df: pd.DataFrame) -> pd.DataFrame:
    '''Generate an evaluation dataframe of NPIs missing episodes
    Args:
        subspecialty_label (`list[str]`): list of subspecialties
        pre_label: label for the original run
        post_label: label for the changed run
        missing_npi_counts_array (`list[list[int]]`) two dimensional array
        containing counts of missing npis by subspecialty index for the pre
        (idx 0) and post (idx 1) runs
        eval_counts_df (`pd.DataFrame`): dataframe of counts created with
        `eval_class_counts`.
    Returns:
        Evaluation dataframe of NPI counts and proportions for those missing
        episodes.
    '''
    # Instantiate dataframe
    return_df = pd.DataFrame()

    # Set up initial structure
    return_df['subspecialty'] = subspecialties
    return_df[pre_label] = missing_npi_counts_array[0]
    return_df[post_label] = missing_npi_counts_array[1]

    # Merge counts df to generate proportions
    return_df = return_df.merge(
        eval_counts_df,
        left_on='subspecialty',
        right_on='category',
        how='inner'
    )

    # Generate proportions
    return_df[f'{pre_label}_prop'] = return_df[pre_label] / \
        return_df[f'{pre_label}_count']
    return_df[f'{post_label}_prop'] = return_df[post_label] / \
        return_df[f'{post_label}_count']

    # Drop counts df columns
    return_df = return_df.drop(
        ['category',
         f'{pre_label}_count',
         f'{post_label}_count',
         'delta'],
        axis=1
    )

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
        logging.error(
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
