import logging

import pandas as pd
from src import evaluate, database

logging.basicConfig(level='INFO',
                    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')

if __name__ == '__main__':

    #######################################
    # Parameters, Paths, & Constants
    #######################################

    path_eval_scores_pre = './cardiology/no_egm_subspecialty/results/train/precision_recall_fscore/precision_recall_fscore_file.xlsx'
    path_eval_scores_post = './cardiology/all_egm_subspecialty/results/train/precision_recall_fscore/precision_recall_fscore_file.xlsx'

    path_cm_pre = './cardiology/no_egm_subspecialty/results/train/confusion_matrix/confusion_matrix_file.xlsx'
    path_cm_post = './cardiology/all_egm_subspecialty/results/train/confusion_matrix/confusion_matrix_file.xlsx'

    path_preds_pre = './cardiology/no_egm_subspecialty/results/propagate/prediction/prediction.xlsx'
    path_preds_post = './cardiology/all_egm_subspecialty/results/propagate/prediction/prediction.xlsx'

    output_dir = './cardiology/evaluation_report/subspecialty_no_all/'
    report_name = 'subspecialty_no_all'

    pre_label = 'no_egm'
    post_label = 'all_egm'

    subspecialties = [
        'Interventional',
        'Electrophysiology',
        'Transplant',
        'Pediatric',
        'ACHD'
    ]

    warehouse = 'LOCAL_BENJAMINWALSH'

    train_data_path = None

    path_to_env_variables = './config/local/credentials.json'

    # sf_connection
    sf_variables = database.load_snowflake_config(
        path_to_env_file=path_to_env_variables,
        warehouse=warehouse
    )
    sf_connection = database.gen_sf_connection(sf_variables)

    #######################################
    # Testing subset evaluation scores
    #######################################
    pre_eval_scores = evaluate.load_evaluation_scores(
        path_eval_scores_pre, pre_label)
    post_eval_scores = evaluate.load_evaluation_scores(
        path_eval_scores_post, post_label)

    compare_eval_scores = evaluate.eval_evaluation_scores(
        pre_eval_scores, post_eval_scores, pre_label=pre_label, post_label=post_label)

    #######################################
    # Confusion matrices
    #######################################
    pre_cm = evaluate.load_confusion_matrix(path_cm_pre)
    post_cm = evaluate.load_confusion_matrix(path_cm_post)

    compare_eval_confusion_matrices = evaluate.eval_confusion_matrices(
        pre_cm,
        post_cm,
        pre_label=pre_label,
        post_label=post_label,
        col_count=len(pre_cm.columns)
    )

    #######################################
    # Class counts
    #######################################

    pre_preds = evaluate.load_predictions(path_preds_pre)
    post_preds = evaluate.load_predictions(path_preds_post)

    pre_class_counts = evaluate.gen_class_counts(pre_preds, pre_label)
    post_class_counts = evaluate.gen_class_counts(post_preds, post_label)

    compare_eval_counts = evaluate.eval_class_counts(
        pre_class_counts,
        post_class_counts,
        pre_label,
        post_label
    )

    #######################################
    # Changed Labels
    #######################################
    changed_labels = evaluate.compare_labels(
        pre_preds,
        post_preds,
        pre_label,
        post_label,
        'y'
    )

    #######################################
    # Static Labels
    #######################################
    static_labels = evaluate.compare_labels(
        pre_preds,
        post_preds,
        pre_label,
        post_label,
        'n'
    )

    #######################################
    # Get a subset of NPIs to check
    #######################################
    changed_labels_to_check = evaluate.gen_switched_label_subset(
        changed_labels,
        train_data_path=train_data_path
    )

    #######################################
    # Probability distributions
    #######################################

    ####################
    # Overall
    ####################

    pre_overall_stats = evaluate.gen_proba_stats(
        pre_preds,
        'probability_1'
    )

    post_overall_stats = evaluate.gen_proba_stats(
        post_preds,
        'probability_1'
    )

    compare_eval_proba_dist_overall = evaluate.eval_proba_distributions(
        pre_overall_stats,
        post_overall_stats,
        pre_label,
        post_label
    )

    ####################
    # Subspecialties
    ####################

    eval_proba_dist_df_list = []

    for subspecialty in subspecialties:

        pre_stats = evaluate.gen_proba_stats(
            pre_preds[pre_preds['label_1'] == subspecialty],
            'probability_1'
        )

        post_stats = evaluate.gen_proba_stats(
            post_preds[post_preds['label_1'] == subspecialty],
            'probability_1'
        )

        compare_eval_proba_dist_subspecialty = evaluate.eval_proba_distributions(
            pre_stats,
            post_stats,
            pre_label,
            post_label
        )

        compare_eval_proba_dist_subspecialty.Name = subspecialty

        eval_proba_dist_df_list.append(
            compare_eval_proba_dist_subspecialty
        )

    ####################
    # Changed Labels
    ####################

    pre_switched_stats = evaluate.gen_proba_stats(
        changed_labels,
        f'{pre_label}_probability_1'
    )

    post_switched_stats = evaluate.gen_proba_stats(
        changed_labels,
        f'{post_label}_probability_1'
    )

    compare_eval_proba_dist_switched = evaluate.eval_proba_distributions(
        pre_switched_stats,
        post_switched_stats,
        pre_label,
        post_label
    )

    #######################################
    # Subspecialty Episode-Level Distribution Analysis
    #######################################

    logging.info('Beginning episode distribution analysis')

    ks_df_list = []
    epi_distribution_df_list = []
    missing_npi_counts = [[], []]

    for subspecialty in subspecialties:

        # Subset predictions to only the subspecialy of interest
        pre_subspecialty = pre_preds[pre_preds['label_1'] == subspecialty]
        post_subspecialty = post_preds[post_preds['label_1'] == subspecialty]
        static_subspecialty = static_labels[static_labels[f'{post_label}_label_1']
                                            == subspecialty]
        changed_subspecialty = changed_labels[changed_labels[f'{post_label}_label_1']
                                              == subspecialty]

        ####################
        # Pre
        ####################

        # Generate the full query for the pre run
        pre_subspecialty_episodes_query = database.gen_episodes_query(
            query=None,  # Default query within the function will handle it
            npis=pre_subspecialty['npi'].to_list()
        )

        # Query the db and generate the full episodes dataframe
        pre_subspecialty_episodes_df = database.gen_episodes_dataframe(
            sf_connection=sf_connection,
            query=pre_subspecialty_episodes_query
        )

        # Pivot the episodes dataframe
        pre_subspecialty_episodes_dummy_df = database.gen_dummy_df_episodes(
            df_episodes=pre_subspecialty_episodes_df
        )

        # Generate the summary stats and restructure the returned df
        pre_subspecialty_episodes_stats = evaluate.gen_summary_stats(
            pre_subspecialty_episodes_dummy_df,
            id_vars=['measure', 'measure_order'],
            var_name='episode'
        )

        # Identify NPIs who go missing because they aren't in the EGM_NPI table
        pre_missing_npis = pd.Series(
            [npi for npi in pre_subspecialty['npi'].to_list(
            ) if npi not in pre_subspecialty_episodes_dummy_df.index]
        )
        pre_missing_npis.name = subspecialty

        logging.debug(f'{subspecialty} (pre) dataframes completed.')

        ####################
        # Post
        ####################

        # Generate the full query for the post run
        post_subspecialty_episodes_query = database.gen_episodes_query(
            query=None,  # Default query within function will handle it
            npis=post_subspecialty['npi'].to_list()
        )

        # Query the db and generate the full episodes dataframe
        post_subspecialty_episodes_df = database.gen_episodes_dataframe(
            sf_connection=sf_connection,
            query=post_subspecialty_episodes_query
        )

        # Pivot the episodes dataframe
        post_subspecialty_episodes_dummy_df = database.gen_dummy_df_episodes(
            df_episodes=post_subspecialty_episodes_df
        )

        # Generate the summary stats and restructure the returned df
        post_subspecialty_episodes_stats = evaluate.gen_summary_stats(
            post_subspecialty_episodes_dummy_df,
            id_vars=['measure', 'measure_order'],
            var_name='episode'
        )

        # Identify NPIs who go missing because they aren't in the EGM_NPI table
        post_missing_npis = pd.Series(
            [npi for npi in post_subspecialty['npi'].to_list(
            ) if npi not in post_subspecialty_episodes_dummy_df.index]
        )
        post_missing_npis.name = subspecialty

        logging.debug(f'{subspecialty} (post) dataframes completed.')

        ####################
        # Static labels
        ####################

        # Generate the full query for the post run
        static_subspecialty_episodes_query = database.gen_episodes_query(
            query=None,  # Default query within function will handle it
            npis=static_subspecialty['npi'].to_list()
        )

        # Query the db and generate the full episodes dataframe
        static_subspecialty_episodes_df = database.gen_episodes_dataframe(
            sf_connection=sf_connection,
            query=static_subspecialty_episodes_query
        )

        # Pivot the episodes dataframe
        static_subspecialty_episodes_dummy_df = database.gen_dummy_df_episodes(
            df_episodes=static_subspecialty_episodes_df
        )

        # Generate the summary stats and restructure the returned df
        static_subspecialty_episodes_stats = evaluate.gen_summary_stats(
            static_subspecialty_episodes_dummy_df,
            id_vars=['measure', 'measure_order'],
            var_name='episode'
        )

        logging.debug(f'{subspecialty} (static) dataframes completed.')

        ####################
        # Changed labels
        ####################

        # Generate the full query for the post run
        changed_subspecialty_episodes_query = database.gen_episodes_query(
            query=None,  # Default query within function will handle it
            npis=changed_subspecialty['npi'].to_list()
        )

        # Query the db and generate the full episodes dataframe
        changed_subspecialty_episodes_df = database.gen_episodes_dataframe(
            sf_connection=sf_connection,
            query=changed_subspecialty_episodes_query
        )

        # Pivot the episodes dataframe
        changed_subspecialty_episodes_dummy_df = database.gen_dummy_df_episodes(
            df_episodes=changed_subspecialty_episodes_df
        )

        # Generate the summary stats and restructure the returned df
        changed_subspecialty_episodes_stats = evaluate.gen_summary_stats(
            changed_subspecialty_episodes_dummy_df,
            id_vars=['measure', 'measure_order'],
            var_name='episode'
        )

        logging.debug(f'{subspecialty} (changed) dataframes completed.')

        # Calculate the KS stats and p-values for each mutual column
        ks_results_subspecialty = evaluate.eval_ks_test(
            pre_subspecialty_episodes_dummy_df,
            post_subspecialty_episodes_dummy_df,
            label_1=pre_label,
            label_2=post_label
        )

        ks_results_subspecialty.Name = subspecialty

        # Compare summary stats
        compare_eval_epi_distributions = evaluate.eval_summary_stats(
            pre_subspecialty_episodes_stats,
            post_subspecialty_episodes_stats,
            pre_label=pre_label,
            post_label=post_label
        )

        # Filter the comparison and reorder based on p-values to make easier to
        # read
        compare_eval_epi_distributions_return = compare_eval_epi_distributions[
            compare_eval_epi_distributions['episode'].isin(
                ks_results_subspecialty['episode']
            )
        ].merge(
            ks_results_subspecialty,
            on='episode',
            how='inner'
        ).sort_values(  # Ensure the most significant difference is first
            by=['p_value', 'measure_order'],
            ascending=[True, True]
        ).drop(  # Drop ks_results columns & measure order
            [
                'df_1',
                'df_2',
                'ks_value',
                'p_value',
                'measure_order'
            ],
            axis=1
        )

        # Merge static and changed label distributions
        compare_eval_epi_distributions_return = evaluate.gen_merged_stats_df(
            compare_eval_epi_distributions_return,
            [
                static_subspecialty_episodes_stats,
                changed_subspecialty_episodes_stats
            ],
            [
                'static',
                'changed'
            ]
        )

        # Add name for easier readability
        compare_eval_epi_distributions_return.Name = subspecialty

        # Add to returned lists of dataframes
        ks_df_list.append(
            ks_results_subspecialty
        )

        epi_distribution_df_list.append(
            compare_eval_epi_distributions_return)

        # Store number of missing npis from episodes
        missing_npi_counts[0].append(
            len(pre_missing_npis)
        )
        missing_npi_counts[1].append(
            len(post_missing_npis)
        )

    # Generate dataframe of missing npi counts

    epi_missing_npis = evaluate.eval_missing_npis(
        subspecialties=subspecialties,
        pre_label=pre_label,
        post_label=post_label,
        missing_npi_counts_array=missing_npi_counts,
        eval_counts_df=compare_eval_counts
    )
    
    #######################################
    # Generate report
    #######################################

    ####################
    # Create dataframe list
    ####################
    comparison_df_list = [
        compare_eval_scores,
        compare_eval_confusion_matrices,
        compare_eval_counts,
        compare_eval_proba_dist_overall,
        compare_eval_proba_dist_switched,
        changed_labels_to_check
    ]

    # Add subspecialty probability distributions
    [
        comparison_df_list.append(
            proba_df
        ) for proba_df in eval_proba_dist_df_list
    ]

    # Add ks and episode distributions
    [
        comparison_df_list.extend(
            [ks_df, stat_df]
        ) for ks_df, stat_df in zip(
            ks_df_list,
            epi_distribution_df_list
        )
    ]

    # Add counts of npis missing episodes
    comparison_df_list.append(epi_missing_npis)

    ####################
    # Create dataframe labels list
    ####################

    comparison_df_labels = [
        'Testing Subset Evaluation Scores',
        'Confusion Matrices',
        'Category Counts',
        'Overall Probability Description',
        'Switched Labels Probability Description',
        'Switched Labels Manual NPI List'
    ]

    # Add subspecialty proba distribution labels
    [
        comparison_df_labels.append(
            proba_df.Name
        ) for proba_df in eval_proba_dist_df_list
    ]

    # Add ks and episode distributions labels
    [
        comparison_df_labels.extend(
            [ks_label.Name,
             epi_distribution_label.Name]
        ) for ks_label, epi_distribution_label in zip(
            ks_df_list,
            epi_distribution_df_list
        )
    ]

    # Add missing npi label
    comparison_df_labels.append(
        'Count of NPIs Missing Episodes by Subspecialty')

    ####################
    # Save report and persist dataframes
    ####################

    evaluate.gen_evaluation_report(
        comparison_df_list,
        comparison_df_labels,
        output_dir=output_dir,
        file_name=report_name
    )
