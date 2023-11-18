import torch
import logging
import hashlib
import json
import numpy as np


def setup_logging(output_dir):
    # Create a logger
    logger = logging.getLogger('gpt-4v-distribution-shift-logger')
    logger.setLevel(logging.INFO)

    # Create a file handler and a console handler
    file_handler = logging.FileHandler(f'{output_dir}/log.txt')
    console_handler = logging.StreamHandler()

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f'Saving in {output_dir}')

    return logger


def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def analyse_unified_output(args):
    logger = logging.getLogger('gpt-4v-distribution-shift-logger')
    # Read data
    data = read_jsonl(
        f'{args.output_dir}/unified_output_{args.model_name}.jsonl')
    # Initialize counters and structures
    total_count = 0
    correct_count = 0
    dataset_domain_class_accuracy = {}

    # Process each entry in the data
    for entry in data:
        dataset = entry['dataset']
        domain = entry['domain']
        true_class = entry['true_class']
        predicted_class = entry['predicted_class']

        # Count for overall accuracy
        total_count += 1
        correct_count += true_class == predicted_class

        # Initialize dataset, domain, and class if not exist
        if dataset not in dataset_domain_class_accuracy:
            dataset_domain_class_accuracy[dataset] = {
                'total': 0, 'correct': 0, 'domains': {}}
        if domain not in dataset_domain_class_accuracy[dataset]['domains']:
            dataset_domain_class_accuracy[dataset]['domains'][domain] = {
                'total': 0, 'correct': 0, 'classes': {}}
        if true_class not in dataset_domain_class_accuracy[dataset]['domains'][domain]['classes']:
            dataset_domain_class_accuracy[dataset]['domains'][domain]['classes'][true_class] = {
                'total': 0, 'correct': 0}

        # Count for dataset, domain, and class accuracy
        dataset_domain_class_accuracy[dataset]['total'] += 1
        dataset_domain_class_accuracy[dataset]['correct'] += true_class == predicted_class
        dataset_domain_class_accuracy[dataset]['domains'][domain]['total'] += 1
        dataset_domain_class_accuracy[dataset]['domains'][domain]['correct'] += true_class == predicted_class
        dataset_domain_class_accuracy[dataset]['domains'][domain]['classes'][true_class]['total'] += 1
        dataset_domain_class_accuracy[dataset]['domains'][domain]['classes'][
            true_class]['correct'] += true_class == predicted_class

    # Calculate overall accuracy
    overall_accuracy = correct_count / total_count if total_count > 0 else 0

    # Calculate accuracy for each dataset, domain, and class
    for dataset, info in dataset_domain_class_accuracy.items():
        info['accuracy'] = info['correct'] / \
            info['total'] if info['total'] > 0 else 0
        for domain, domain_info in info['domains'].items():
            domain_info['accuracy'] = domain_info['correct'] / \
                domain_info['total'] if domain_info['total'] > 0 else 0
            for class_name, class_info in domain_info['classes'].items():
                class_info['accuracy'] = class_info['correct'] / \
                    class_info['total'] if class_info['total'] > 0 else 0

    # Save results to JSON file
    output_data = {
        'overall_accuracy': overall_accuracy,
        'datasets': dataset_domain_class_accuracy
    }
    with open(f'{args.output_dir}/results_{args.model_name}.json', 'w') as outfile:
        json.dump(output_data, outfile, indent=4)
    logger.info(
        f'saving results into {args.output_dir}/results_{args.model_name}.json')


# def seed_hash(*args):
#     """
#     Derive an integer hash from all args, for use as a random seed.
#     """
#     args_str = str(args)
#     return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


# class _SplitDataset(torch.utils.data.Dataset):
#     """Used by split_dataset"""

#     def __init__(self, underlying_dataset, keys):
#         super(_SplitDataset, self).__init__()
#         self.underlying_dataset = underlying_dataset
#         self.keys = keys

#     def __getitem__(self, key):
#         return self.underlying_dataset[self.keys[key]]

#     def __len__(self):
#         return len(self.keys)


# def split_dataset(dataset, n, seed=0):
#     """
#     Return a pair of datasets corresponding to a random split of the given
#     dataset, with n datapoints in the first dataset and the rest in the last,
#     using the given random seed
#     """
#     assert (n <= len(dataset))
#     keys = list(range(len(dataset)))
#     np.random.RandomState(seed).shuffle(keys)
#     keys_1 = keys[:n]
#     keys_2 = keys[n:]
#     return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


# class _InfiniteSampler(torch.utils.data.Sampler):
#     """Wraps another Sampler to yield an infinite stream."""

#     def __init__(self, sampler):
#         self.sampler = sampler

#     def __iter__(self):
#         while True:
#             for batch in self.sampler:
#                 yield batch


# class InfiniteDataLoader:
#     def __init__(self, dataset, batch_size, num_workers, weights=None):
#         super().__init__()
#         self._length = len(dataset) / batch_size

#         if weights is not None:
#             sampler = torch.utils.data.WeightedRandomSampler(weights,
#                                                              replacement=True,
#                                                              num_samples=batch_size)
#         else:
#             sampler = torch.utils.data.RandomSampler(dataset,
#                                                      replacement=True)

#         if weights == None:
#             weights = torch.ones(len(dataset))

#         batch_sampler = torch.utils.data.BatchSampler(
#             sampler,
#             batch_size=batch_size,
#             drop_last=True)

#         self._infinite_iterator = iter(torch.utils.data.DataLoader(
#             dataset,
#             num_workers=num_workers,
#             batch_sampler=_InfiniteSampler(batch_sampler)
#         ))

#     def __iter__(self):
#         while True:
#             yield next(self._infinite_iterator)

#     def __len__(self):
#         raise ValueError
