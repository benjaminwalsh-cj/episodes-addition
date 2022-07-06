from src import evaluate


if __name__ == '__main__':
    # Testing subset evaluation scores
    path_eval_scores_pre = './no_egm/results/train/precision_recall_fscore/precision_recall_fscore_file.xlsx'
    path_eval_scores_post = './all_egm/results/train/precision_recall_fscore/precision_recall_fscore_file.xlsx'

    pre_eval_scores = evaluate.load_evaluation_scores(
        path_eval_scores_pre, 'no_egm')
    post_eval_scores = evaluate.load_evaluation_scores(
        path_eval_scores_post, 'all_egm')

    compare_eval_scores = evaluate.eval_evaluation_scores(
        pre_eval_scores, post_eval_scores, pre_label='no_egm', post_label='all_egm')

    # Class counts
    path_preds_pre = './no_egm/results/propagate/prediction/prediction.xlsx'
    path_preds_post = './all_egm/results/propagate/prediction/prediction.xlsx'

    pre_preds = evaluate.load_predictions(path_preds_pre)
    post_preds = evaluate.load_predictions(path_preds_post)

    pre_class_counts = evaluate.gen_class_counts(pre_preds, 'no_egm')
    post_class_counts = evaluate.gen_class_counts(post_preds, 'all_egm')

    compare_eval_counts = evaluate.eval_class_counts(
        pre_class_counts,
        post_class_counts,
        'no_egm',
        'all_egm'
    )

    # Changed Labels
    changed_labels = evaluate.id_changed_labels(
        pre_preds,
        post_preds,
        'no_egm',
        'all_egm'
    )

    # Get a subset of NPIs to check
    changed_labels_to_check = evaluate.gen_switched_label_subset(
        changed_labels,
        train_data_path='./all_egm/joblib/data.joblib'
    )

    # Probability distributions

    # Overall
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
        'no_egm',
        'all_egm'
    )

    ## Classes - Cardiology
    pre_cardio_stats = evaluate.gen_proba_stats(
        pre_preds[pre_preds['label_1'] == 'Cardiology'],
        'probability_1'
    )

    post_cardio_stats = evaluate.gen_proba_stats(
        post_preds[pre_preds['label_1'] == 'Cardiology'],
        'probability_1'
    )

    compare_eval_proba_dist_cardio = evaluate.eval_proba_distributions(
        pre_cardio_stats,
        post_cardio_stats,
        'no_egm',
        'all_egm'
    )

    # Changed Labels
    pre_switched_stats = evaluate.gen_proba_stats(
        changed_labels,
        'no_egm_probability_1'
    )

    post_switched_stats = evaluate.gen_proba_stats(
        changed_labels,
        'all_egm_probability_1'
    )

    compare_eval_proba_dist_switched = evaluate.eval_proba_distributions(
        pre_switched_stats,
        post_switched_stats,
        'no_egm',
        'all_egm'
    )

    # Generate report
    comparison_df_list = [
        compare_eval_scores,
        compare_eval_counts,
        compare_eval_proba_dist_overall,
        compare_eval_proba_dist_cardio,
        compare_eval_proba_dist_switched,
        changed_labels_to_check
    ]

    comparison_df_labels = [
        'Testing Subset Evaluation Scores',
        'Category Counts',
        'Overall Probability Description',
        'Cardiology Labeled Probability Description',
        'Switched Labels Probability Description',
        'Switched Labels Manual NPI List'
    ]

    evaluate.gen_evaluation_report(
        comparison_df_list,
        comparison_df_labels,
        output_dir='./evaluation_report/',
        file_name='evaluation_report'
    )
