from typing import Tuple, List, Optional
import string
import re
import logging
import argparse
import os
from collections import defaultdict

from transliterate import translit
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UrlTokenizer:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
    
    def _get_words(self, url) -> Tuple[str, List[str]]:
        return get_words(url)
    
    def tokenize_with_status(self, url: str) -> Tuple[str, torch.Tensor]:
        status, words = self._get_words(url)
        words_str = ' '.join(words)
        tokenized = self.tokenizer(words_str, return_tensors="pt")
        return status, tokenized
    
    def tokenize(self, url: str) -> torch.Tensor:
        return self.tokenize_with_status(url)[1]
    
    def __call__(self, url: str) -> torch.Tensor:
        return self.tokenize(url)


#############  Seq Embedders  #############

class SeqEmbedderBase(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
    
    def embed(self, tokenized: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def __call__(self, tokenized: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.embed(tokenized, attention_mask)
    

class MeanPoolSeqEmbedder(SeqEmbedderBase):
    def embed(self, tokenized: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        model_output = self.model(tokenized, attention_mask)
        return mean_pooling(model_output, attention_mask)
    

class EmbedderPoolerOutput(SeqEmbedderBase):
    def embed(self, tokenized: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        model_output = self.model(tokenized, attention_mask)
        return model_output['pooler_output']


class EmbedderCLS(SeqEmbedderBase):
    def embed(self, tokenized: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        model_output = self.model(tokenized, attention_mask)
        cls_pos = 0
        return model_output['last_hidden_state'][:, cls_pos, :]


############ Data Loading utils ############

class UrlDataset(Dataset):
    def __init__(self, urls: List[str], tokenizer) -> None:
        self.urls = urls
        self.tokenizer = UrlTokenizer(tokenizer)
    
    def __len__(self) -> int:
        return len(self.urls)
    
    def __getitem__(self, idx) -> torch.Tensor:
        url = self.urls[idx]
        input_ids = self.tokenizer(url)['input_ids'].squeeze(0)  # Squeeze to remove batch dimension
        mask = self.tokenizer(url)['attention_mask'].squeeze(0)
        return input_ids, mask

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)  # Padding with 0
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)  # Padding with 0
    
    return padded_input_ids, padded_attention_masks


############################
 
def is_punycode(s: str) -> bool:
    return s.startswith('xn--')    
    

def convert_punycode(puny_domain: str) -> str:
    """
    Converts punycode to unicode

    Example:
    >>> convert_punycode('xn--22-glcqfm3bya1b.xn--p1ai')
    <<< 'грузчик22.рф'
    """
    return puny_domain.encode().decode('idna')


def remove_extension(url: str) -> str:
    # I removed `ru-an` specifically, because it was the only
    # subdomain present in mixed cyrillic-latin domains. 
    # Without `ru-an` there are no mixed domains like коронавирус.ru-an.info
    
    #  Нужно ли удалять:
    # livejournal.com
    # turbopages.org
    
    url = ('.').join(url.split('.')[:-1])
    url = url[:-6] if url.endswith('ru-an') else url
    return url


def get_mixed_lang_domains(url_hosts: list) -> set:
    # To get knowledge about urls that contain 
    # both cyrillic and latin characters.
    mixed_lang_domains = set()

    for url in url_hosts:
        if is_punycode(url):
            url = convert_punycode(url)
        url = remove_extension(url)
        url_chars_set = set(url)
        
        if url_chars_set - LATIN_CHARS != url_chars_set and url_chars_set - CYRILLIC_CHARS != url_chars_set:
            # url has both cyrillic and latin
            mixed_lang_domains.add(url)
    return mixed_lang_domains


def convert_to_cyrillic(url: str) -> Tuple[str, str]:
    statuses = []
    url_set = set(url)
    is_latin =  (url_set - LATIN_CHARS) != url_set
    is_cyrillic =  (url_set - CYRILLIC_CHARS) != url_set 
    if is_latin:
#         assert not is_cyrillic, f"Встречен домен содержащий и латиницу и кириллицу: {url}"
        url = translit(url, 'ru')
    
        if is_cyrillic:
            statuses.append("transliterated. cyr_and_latin_before_translit")
            
        url_set = set(url)
        
        if ((url_set - CYRILLIC_CHARS) == url_set):
            statuses.append("no_cyrillic_after_translit")
        has_xwq = url_set.union({'x'}) == url_set or url_set.union({'w'}) == url_set or url_set.union({'q'}) == url_set
        if has_xwq:
            statuses.append("has_xwq")
        elif (url_set - LATIN_CHARS) != (url_set):
            statuses.append("unexpected_eng_left")
        else:
            statuses.append("proper_translit")
    elif is_cyrillic:
        statuses.append('initially_ru')
    else:
        statuses.append("no_eng_no_ru")
    status = ' '.join(statuses)
    return status, url


def get_words(url: str) -> Tuple[str, List[str]]:
    orig_url = url
    if is_punycode(url):
        url = convert_punycode(url)
    url = remove_extension(url)
    status, url = convert_to_cyrillic(url)
    words = re.findall(r'[а-я]+', url)
    return status, words


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-10)
    return sum_embeddings / sum_mask


def get_output_file_name(input_file_path: str, model_name: str, extension: str) -> str:
    input_file_name = os.path.basename(input_file_path)
    model_name = model_name.replace('/', '_')
    return f'{input_file_name}_{model_name}.{extension}'


def save(data: Tuple[List[str], torch.Tensor], output_file_path: str):
    urls, embeddings = data
    embeddings_list = embeddings.tolist()
    data = {'url_host': urls, 'url_host_embedding': embeddings_list}
    df = pd.DataFrame(data)
    df.to_parquet(output_file_path, engine='pyarrow', index=False)
    print(f"Parquet file saved to {output_file_path}")


def get_device(device_str: Optional[str]) -> torch.device:
    if device_str is not None:
        assert device_str in ['cpu', 'cuda'], f'Incorrect device: {device_str}'
        device = torch.device(device_str)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls-file-path', type=os.path.abspath,
                        help = 'A path to a file utf-8 file with urls. ' \
                                'Each url is in a new line.')
    parser.add_argument('--model-name', type=str, default="sberbank-ai/ruElectra-small",
                        help = 'A name of the hugingface model to use for tokenization. ' \
                                'Default is "ai-forever/ruBert-base"')
    parser.add_argument('--batch-size', type=int, default=10000,
                        help = 'A batch size for the dataloader. Default is 10000')
    parser.add_argument('--num-workers', type=int, default=0,
                        help = 'A number of workers for the dataloader. Default is 0')
    parser.add_argument('--device', type=str, default=None,
                        help = 'A device to use for the model. Default is "cuda" if available, else "cpu"')
    parser.add_argument('--output-file-dir', type=os.path.abspath,
                        help = 'A path to a file where to save the results')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
 
    LATIN_CHARS = set(string.ascii_lowercase)
    DIGITS = set(string.digits)
    SPECIAL_CHARS = set('.-_')
    CYRILLIC_CHARS = set('абвгдежзийклмнопрстуфхцчшщъыьэюя')  # кроме ё, потому что не встречается

    POSSIBLE_CHARS = set.union(LATIN_CHARS, DIGITS, SPECIAL_CHARS, CYRILLIC_CHARS)



    args = parse_args()


    # args = argparse.Namespace(
    #     urls_file_path = '/kaggle/input/mts-urls/urls.txt',
    #     model_name = 'sberbank-ai/ruElectra-small',
    #     batch_size= 10000,
    #     num_workers = 0,
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu',
    #     output_file_dir = '/kaggle/working/'
    # )



    device = get_device(args.device)
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    embedder = MeanPoolSeqEmbedder(model).to(device)

    logger.info(f'possible chars: {POSSIBLE_CHARS}')

    with open(args.urls_file_path, encoding = 'utf-8') as f:
        url_hosts = f.read().splitlines()

    chars_met = {c for s in url_hosts for c in s}
    # <= operator on sets checks if the left set is a subset of the right set
    assert chars_met <= POSSIBLE_CHARS, f'Unexpected characters met: {chars_met - POSSIBLE_CHARS}'

    if chars_met == POSSIBLE_CHARS:
        logger.info('All possible characters met, no impossible characters met')


    mixed_lang_domains = get_mixed_lang_domains(url_hosts)

    logger.info(f'Mixed language domains number: {len(mixed_lang_domains)}')
    logger.info(f'Mixed language domains: {mixed_lang_domains}')

    status_to_urls = defaultdict(list)

    for url in tqdm(url_hosts):
        orig_url = url
        status, words = get_words(url)
        status_to_urls[status].append(orig_url)


    status_to_n_urls = {k: len(v) for k, v in status_to_urls.items()}

    logger.info(f'Statuses to number of urls: {status_to_n_urls}')

    n_urls = 10
    for status, urls in status_to_urls.items():
        print(f"{status}:")
        print("-----")
        urls = urls[:n_urls]
        words_lsts = [get_words(url)[1] for url in urls]
        for url in urls[:n_urls]:
            print(url)
            if is_punycode(url):
                print(f"after puny_code conversion: {convert_punycode(url)}")
            print(get_words(url)[1])
        print()


    url_dataset = UrlDataset(url_hosts, hf_tokenizer)

    url_dataloader = DataLoader(url_dataset, batch_size=args.batch_size, 
                                shuffle=False, collate_fn = collate_fn,
                                num_workers=args.num_workers)


    all_outputs = []

    with torch.inference_mode():
        for input_ids, attention_masks in tqdm(url_dataloader):
            input_ids=input_ids.to('cuda')
            attention_masks=attention_masks.to('cuda')
            sentence_embeddings = embedder(tokenized = input_ids, attention_mask = attention_masks) 
            all_outputs.append(sentence_embeddings.to('cpu'))

    all_embeddings = torch.cat(all_outputs, dim = 0)

    assert len(all_embeddings) == len(url_hosts)



    embeddings_list = all_embeddings.tolist()

    data = {'url_host': url_hosts, 'url_host_embedding': embeddings_list}
    df = pd.DataFrame(data)


    out_fname = get_output_file_name(input_file_path = args.urls_file_path, model_name = args.model_name, extension = 'parquet')
    output_path = os.path.join(args.output_file_dir, out_fname)
    df.to_parquet(output_path, engine='pyarrow', index=False)

    print(f"Parquet file saved to {output_path}")