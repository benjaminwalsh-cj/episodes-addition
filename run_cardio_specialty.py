from src import evaluate


if __name__ == '__main__':

    # Parameters & Paths
    path_eval_scores_pre = './cardiology/no_egm/results/train/precision_recall_fscore/precision_recall_fscore_file.xlsx'
    path_eval_scores_post = './cardiology/tx_egm/results/train/precision_recall_fscore/precision_recall_fscore_file.xlsx'

    path_cm_pre = './cardiology/no_egm/results/train/confusion_matrix/confusion_matrix_file.xlsx'
    path_cm_post = './cardiology/tx_egm/results/train/confusion_matrix/confusion_matrix_file.xlsx'

    path_preds_pre = './cardiology/no_egm/results/propagate/prediction/prediction.xlsx'
    path_preds_post = './cardiology/tx_egm/results/propagate/prediction/prediction.xlsx'

    output_dir = './cardiology/evaluation_report/specialty_no_tx/'
    report_name = 'specialty_no_tx'

    pre_label = 'no_egm'
    post_label = 'tx_egm'

    train_data_path = None

    # Testing subset evaluation scores
    pre_eval_scores = evaluate.load_evaluation_scores(
        path_eval_scores_pre, pre_label)
    post_eval_scores = evaluate.load_evaluation_scores(
        path_eval_scores_post, post_label)

    compare_eval_scores = evaluate.eval_evaluation_scores(
        pre_eval_scores, post_eval_scores, pre_label=pre_label, post_label=post_label)

    # Confusion matrices
    pre_cm = evaluate.load_confusion_matrix(path_cm_pre)
    post_cm = evaluate.load_confusion_matrix(path_cm_post)

    compare_eval_confusion_matrices = evaluate.eval_confusion_matrices(
        pre_cm,
        post_cm,
        pre_label=pre_label,
        post_label=post_label,
        col_count=len(pre_cm.columns)
    )

    # Class counts

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

    # Changed Labels
    changed_labels = evaluate.compare_labels(
        pre_preds,
        post_preds,
        pre_label,
        post_label,
        'y'
    )

    # Get a subset of NPIs to check
    changed_labels_to_check = evaluate.gen_switched_label_subset(
        changed_labels,
        train_data_path=train_data_path
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
        pre_label,
        post_label
    )

    # Classes - Cardiology
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
        pre_label,
        post_label
    )

    # Changed Labels
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

    # Generate report
    comparison_df_list = [
        compare_eval_scores,
        compare_eval_confusion_matrices,
        compare_eval_counts,
        compare_eval_proba_dist_overall,
        compare_eval_proba_dist_cardio,
        compare_eval_proba_dist_switched,
        changed_labels_to_check
    ]

    comparison_df_labels = [
        'Testing Subset Evaluation Scores',
        'Confusion Matrix',
        'Category Counts',
        'Overall Probability Description',
        'Cardiology Labeled Probability Description',
        'Switched Labels Probability Description',
        'Switched Labels Manual NPI List'
    ]

    evaluate.gen_evaluation_report(
        comparison_df_list,
        comparison_df_labels,
        output_dir=output_dir,
        file_name=report_name
    )
