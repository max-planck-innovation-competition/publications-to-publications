"""Encode publication titles and abstracts for citation ranking."""
import json
import os
import logging
from sentence_transformers import (SentenceTransformer,
                                   models)
from src import config
from src.models.reranking_evaluator import evaluate_ranking


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def read_jsonl(dataset_path):
    """Yield JSON objects directly from a local JSONL file."""
    with open(dataset_path, encoding='utf-8') as data_file:
        for line_number, line in enumerate(data_file, start=1):
            if not line.strip():
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {dataset_path}"
                ) from error


def load_test_data(dataset_path: str, sep_token: str = ' [SEP] ', q_text: str = None, doc_text: str = None):
    """
    concatenate title and abstract of query, positive and negative samples, using sep_token, particularly for TEST data
    Parameters
    ----------
    dataset_path: str (path to test dataset)
    sep_token: str = ' [SEP] '
    q_text: str = None (prefix to add to query text)
    doc_text: str = None (prefix to add to positive and negative texts)

    """
    logger.info('loading TEST data from {}'.format(dataset_path))
    items = []
    for example in read_jsonl(dataset_path):
        sample = {
            'positive': [],
            'negative': []
        }
        query = (example['query']['title'] or " ") + \
            sep_token + (example['query']['abstract'] or " ")
        if q_text:
            query = q_text + query
        sample['query'] = query

        for pos in example['pos']:
            pos = (pos['title'] or " ") + sep_token + (pos['abstract'] or " ")
            if doc_text:
                pos = doc_text + pos
            sample['positive'].append(pos)

        for neg in example['neg']:
            neg = (neg['title'] or " ") + sep_token + (neg['abstract'] or " ")
            if doc_text:
                neg = doc_text + neg
            sample['negative'].append(neg)

        items.append(sample)

    logger.info('loaded %d test queries', len(items))
    return items


def test_model_with_ranking(model_path: str,
                            test_dataset_path: str,
                            test_data_type: str,
                            pooling_mode: str = 'cls',
                            q_text: str = None,
                            doc_text: str = None,
                            normalize: bool = False,
                            adapter: str = None):
    """
    Rank-aware evaluation
    """
    # get model name from path (last part of the path)
    model_name = os.path.basename(model_path)
    logger.info("test %s with ranking and pooling mode=%s", model_name,
                pooling_mode)
    transformer = models.Transformer(
        model_path
    )
    if adapter:
        # Load the adapter on top of the existing base model.
        import adapters
        adapters.init(transformer.auto_model)
        transformer.auto_model.load_adapter(adapter, source="hf", set_active=True)

    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode=pooling_mode
    )

    if normalize:
        # Add normalization module

        modules = [transformer, pooling, models.Normalize()]
    else:
        modules = [transformer, pooling]

    model = SentenceTransformer(modules=modules)

    # concatenate title and abstract
    data_test = load_test_data(test_dataset_path, q_text=q_text, doc_text=doc_text)

    OUTPUT_PATH: str = f"{config.Config.OUTPUT_PATH}/{test_data_type}/ranking/{pooling_mode}"

    result = evaluate_ranking(
        model,
        data_test,
        output_path=OUTPUT_PATH,
        model_name=model_name,
        show_progress_bar=True,
    )
    print(result)
