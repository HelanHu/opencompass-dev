from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator, RougeEvaluator, SquadEvaluator
from opencompass.datasets import LEvalTVShowSummDataset

LEval_tvshow_summ_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answer',
    train_split='test',
    test_split='test'
)

LEval_tvshow_summ_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{context}\nQuestion: {question}'),
                dict(role='BOT', prompt='TL;DR:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)

LEval_tvshow_summ_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator), 
    pred_role='BOT'
)

LEval_tvshow_summ_datasets = [
    dict(
        type=LEvalTVShowSummDataset,
        abbr='LEval_tvshow_summ',
        path='L4NLP/LEval',
        name='tv_show_summ',
        reader_cfg=LEval_tvshow_summ_reader_cfg,
        infer_cfg=LEval_tvshow_summ_infer_cfg,
        eval_cfg=LEval_tvshow_summ_eval_cfg)
]
