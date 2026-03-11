from .fast_mapper import LemmatizerBMP
from .abstract_mapper import AbstractLemmatizer
from torch import nn, Tensor
import torch
from typing import Dict, List, Literal, Optional, Tuple, Union

alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
default_phoc_mapper = LemmatizerBMP.from_alphabet_mapping({c:c.lower() for c in alpha}, unknown_chr="�")


class PHOC(nn.Module):
    """Pyramidal Histogram of Characters (PHOC) representation for word spotting.
    
    This class implements the PHOC descriptor for encoding text sequences into
    fixed-size feature vectors, commonly used in word spotting and recognition tasks.
    
    References
    ----------
    .. [1] Sudholt, S., & Fink, G. A. (2016). PHOCNet: A deep convolutional neural
           network for word spotting in handwritten documents. In 2016 15th
           International Conference on Frontiers in Handwriting Recognition (ICFHR)
           (pp. 277-282). IEEE.
           https://arxiv.org/pdf/1604.00187
    
    .. [2] Almazán, J., Gordo, A., Fornés, A., & Valveny, E. (2014). Word spotting
           and recognition with embedded attributes. IEEE transactions on pattern
           analysis and machine intelligence, 36(12), 2552-2566.
           http://pages.cvc.uab.es/afornes/publi/journals/2014_PAMI_Almazan.pdf
    
    Parameters
    ----------
    levels : List[int], optional
        Pyramid levels for histogram pooling, by default [1, 2, 3, 5]
    mapper : AbstractLemmatizer, optional
        Character mapper for string-to-onehot encoding, by default LemmatizerBMP initialized with case insensitive alphanumeric characters
    normalization_mode : {'avg', 'sum', 'bin'}, optional
        Normalization mode for histograms, by default 'avg'
    """
    

    def __init__(self, levels: List[int]=[1,2,3,5], mapper: AbstractLemmatizer = default_phoc_mapper,
                 normalization_mode: Literal["avg", "sum", "bin"]= "avg"):
        super().__init__()
        self.levels = levels
        self.mapper = mapper
        self.pool_layers = nn.ModuleList()
        for level in levels:
            self.pool_layers.append(nn.AdaptiveAvgPool1d(level))
        if normalization_mode == "avg":
            self.denormalise = False
            self.chop_bins = False
        elif normalization_mode == "sum":
            self.denormalise = True
            self.chop_bins = False
        elif normalization_mode == "bin":
            self.denormalise = True
            self.chop_bins = True
        else:
            raise ValueError(f"Unknown PHOC mode: {normalization_mode}")


    def normalization_mode(self)-> str:
        """Get the current normalization mode.
        
        Returns
        -------
        str
            The normalization mode: 'avg', 'sum', or 'bin'
        """
        if self.denormalise and self.chop_bins:
            return "bin"
        elif self.denormalise and not self.chop_bins:
            return "sum"
        else:
            return "avg"
    
    def get_phoc_sz(self)-> int:
        """Get the size of the PHOC feature vector.
        
        Returns
        -------
        int
            Total dimension of PHOC representation (sum of levels * character space)
        """
        return sum(self.levels) * len(self.mapper)


    def encode_sequence(self, tc_seq: Tensor)-> Tensor:
        """Encode a single sequence into PHOC representation.
        
        Parameters
        ----------
        tc_seq : Tensor
            Sequence tensor of shape (T, C) where T is sequence length
            and C is character dimension
        
        Returns
        -------
        Tensor
            PHOC representation of shape (1, sum(levels)*C)
        """
        assert tc_seq.dim() == 2  # (T, C)
        btc_seq = tc_seq.unsqueeze(0)  # (1, T, C)
        bct_seq = btc_seq.permute(0,2,1)  # (1, C, T)
        pooled = []
        for pool_layer in self.pool_layers:
            bch = pool_layer(bct_seq).transpose(1,2)#.reshape(1, -1)
            if self.denormalise:
                bch = bch * tc_seq.size(0)
            if self.chop_bins:
                bch[bch > 1.0] = 1.0
            histograms = bch.reshape(1, -1)  # (1, C*L)
            pooled.append(histograms)  # (1, C*L)
        phoc_rep = torch.cat(pooled, dim=1) # (1, sum L)
        return phoc_rep
    

    def encode_batch(self, btc_seqs: Tensor, seq_lengths: Optional[Tensor])-> Tensor:
        """Encode a batch of sequences into PHOC representations.
        
        Parameters
        ----------
        btc_seqs : Tensor
            Batch tensor of shape (B, T_max, C) where B is batch size,
            T_max is maximum sequence length, and C is character dimension
        seq_lengths : Tensor, optional
            Actual lengths of sequences in batch, shape (B,).
            If None, all sequences are assumed full length
        
        Returns
        -------
        Tensor
            PHOC representations of shape (B, sum(levels)*C)
        """
        res = []
        if seq_lengths is None:
            seq_lengths = torch.full((btc_seqs.shape[0],), btc_seqs.shape[1], dtype=torch.long, device=btc_seqs.device)
        for n in range(len(seq_lengths)):
            tc_seq = btc_seqs[n, :seq_lengths[n], :]  # (T, C)
            phoc_rep = self.encode_sequence(tc_seq)    # (sum L, C)
            res.append(phoc_rep)
        return torch.stack(res, dim=0)  # (B, sum L)
    

    def encode_string(self, s: str)-> Tensor:
        """Encode a single string into PHOC representation.
        
        Parameters
        ----------
        s : str
            Input string to encode
        
        Returns
        -------
        Tensor
            PHOC representation of shape (1, sum(levels)*C)
        """
        tc_seq = self.mapper.str_to_onehot(s)  # (T, C)
        tc_seq = Tensor(tc_seq)
        phoc_rep = self.encode_sequence(tc_seq)           # (1, sum L)
        return phoc_rep


    def encode_string_list(self, s: List[str])-> Tensor:
        """Encode a list of strings into PHOC representations.
        
        Parameters
        ----------
        s : List[str]
            List of strings to encode
        
        Returns
        -------
        Tensor
            PHOC representations of shape (B, sum(levels)*C) where B is number of strings
        """
        res = []
        for str_ in s:
            res.append(self.encode_string(str_))
        return torch.stack(res, dim=0)  # (B, sum L)
    

    def forward(self, seqs: Union[Tensor, str, List[str]], seq_lengths: Optional[Tensor] = None)-> Tensor:
        """Forward pass supporting multiple input types.
        
        Parameters
        ----------
        seqs : Tensor, str, or List[str]
            Input sequences. Can be:
            - Tensor of shape (B, T, C) for batch encoding
            - Tensor of shape (T, C) for single sequence encoding
            - str for single string
            - List[str] for multiple strings
        seq_lengths : Tensor, optional
            Sequence lengths for batch input, shape (B,)
        
        Returns
        -------
        Tensor
            PHOC representations
        
        Raises
        ------
        ValueError
            If input type is not supported
        """
        if isinstance(seqs, Tensor) and seqs.dim() == 3:
            return self.encode_batch(seqs, seq_lengths)
        elif isinstance(seqs, Tensor) and seqs.dim() == 2:
            return self.encode_sequence(seqs)
        elif isinstance(seqs, str):
            return self.encode_string(seqs)
        elif isinstance(seqs, list):
            return self.encode_string_list(seqs)
        else:
            raise ValueError("Invalid input type for PHOC encoding")


class SemanticEmbeddings(torch.nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int=256, stoi: Optional[Dict[str, int]]=None, **kwargs):
        if "padding_idx" not in kwargs:
            kwargs["padding_idx"] = 0
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim, **kwargs)
        if stoi is None:
            self.stoi = {str(n): n for n in range(num_embeddings)}
        self.itos: Dict[int, str] = {i:s for i, s in self.stoi.items()}
        self.stoi["<PAD>"] = kwargs["padding_idx"]
        self.itos[kwargs["padding_idx"]] = "<PAD>"
    
    def forward(self, seqs: Tensor)-> Tensor:
        return super().forward(seqs)
    
    def get_embedding(self, word: Union[str, int])-> Tensor:
        if isinstance(word, str):
            idx = self.stoi[word]
        elif isinstance(word, int):
            idx = word
        else:
            raise ValueError("word must be str or int")
        return self.weight[idx, :]
    
    def get_embedings(self, words: List[Union[str, int]])-> Tensor:
        idxs = []
        for word in words:
            if isinstance(word, str):
                idxs.append(self.stoi[word])
            elif isinstance(word, int):
                idxs.append(word)
            else:
                raise ValueError("word must be str or int")
        idxs_tensor = torch.tensor(idxs, device=self.weight.device, dtype=torch.long)
        return self.weight[idxs_tensor, :]

    def get_distances(self, embeddings: Tensor)-> Tensor:
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)  # (1, D)
        assert embeddings.dim() == 2
        dists = self.weight @ embeddings.t()  # (N, 1)
        norm_dm = torch.norm(self.weight, dim=1, keepdim=True)  # (N, 1)
        norm_query = torch.norm(embeddings, dim=1, keepdim=True)  # (1, B)
        dists = dists / (norm_dm * norm_query.t() + 1e-8)  # (N, B)
        return dists
    
    def retrieve_topk_similar(self, embeddings: Tensor, k: int = 1)-> Tuple[Tensor, Tensor, Tensor]:
        distances_db_q = self.get_distances(embeddings)  # (DB SZ, NB QUERIES)
        similarities_rank_q, indexes_rank_q = torch.topk(distances_db_q, k=k, dim=0, largest=True)  # (k, NB QUERIES)
        res_query_k_emdedding = self.weight[indexes_rank_q, :]  # (k, NB QUERIES, D)
        return similarities_rank_q, indexes_rank_q, res_query_k_emdedding

    def get_closest_word(self, embedding: Tensor)-> str:
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # (1, D)
        dists = torch.norm(self.weight - embedding, dim=1)  # (N,)
        closest_idx = torch.argmin(dists).item()
        return self.itos[closest_idx]
    
    def get_word_ranks(self, word1: Union[str, int], word2: Union[str, int])-> Tuple[float, float, int]:
        raise NotImplementedError("get_word_ranks is not implemented yet")


class PHOCDictionary:
    def __init__(self, phoc: PHOC, max_size: int = 10000):
        self.phoc = phoc
        self.vocab_size = max_size
        self.nb_words = 0
        self.stoi: Dict[str, int] = {}  # char to index
        self.itos: Dict[int, str] = {}  # index to char
        self.word_freq: Dict[str, int] = {}
        self.database_phocs = torch.zeros((max_size, phoc.get_phoc_sz()))  # (N, D)
    
    def add_word(self, word: str, replace_if_needed: bool = False):
        if word in self.word_freq:
            self.word_freq[word] += 1
        else:
            self.word_freq[word] = 1
        if word in self.stoi:
            return
        if self.nb_words >= self.vocab_size:
            if replace_if_needed:
                # Find least frequent word
                least_frequent_word = min(self.word_freq, key=self.word_freq.get)
                index = self.stoi[least_frequent_word]
                # Remove least frequent word
                del self.stoi[least_frequent_word]
                del self.itos[index]
                del self.word_freq[least_frequent_word]
                # Add new word at the same index
                self.stoi[word] = index
                self.itos[index] = word
                self.word_freq[word] = 1
                phoc_rep = self.phoc.encode_string(word).squeeze(0)  # (D,)
                self.database_phocs[index, :] = phoc_rep
            raise ValueError("PHOCDictionary max size exceeded")
        self.stoi[word] = self.nb_words
        self.itos[self.nb_words] = word
        self.word_freq[word] = 1
        phoc_rep = self.phoc.encode_string(word).squeeze(0)  # (D,)
        self.database_phocs[self.nb_words, :] = phoc_rep
        self.nb_words += 1