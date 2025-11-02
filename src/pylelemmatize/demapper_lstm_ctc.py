import torch
from tqdm import tqdm

from pylelemmatize.mapper_ds import Seq2SeqDs
from .fast_mapper import LemmatizerBMP
from typing import Any, Union, List, Tuple, Dict, Literal, Optional


class DemapperLSTMCTC(torch.nn.Module):
    def __init__(self, input_alphabet: str, output_alphabet: str, hidden_sizes: List[int]=[128, 128, 128], 
                 dropouts: Union[List[float], float] = 0., directions: Union[Literal[-1, 0, 1], List[Literal[-1, 0, 1]]] = 0):
        super(DemapperLSTMCTC, self).__init__()
        assert all([sz%2 == 0 for sz in hidden_sizes]), f"All hidden sizes must be even numbers. {hidden_sizes}"
        self.input_mapper = LemmatizerBMP(mapping_dict={c: c for c in input_alphabet}, unknown_chr='�')
        self.output_mapper = LemmatizerBMP(mapping_dict={c: c for c in output_alphabet}, unknown_chr='�')
        self.input_embedding = torch.nn.Embedding(len(input_alphabet) + 1, hidden_sizes[0])
        hidden_sizes = hidden_sizes + [len(self.output_mapper)]  # Add output alphabet size as the last layer size

        if isinstance(directions, int):
            assert directions in [-1, 0, 1], "directions must be -1, 0 or 1"
            directions = [directions] * (len(hidden_sizes) - 1)
        elif isinstance(directions, list):
            assert len(directions) == len(hidden_sizes) - 1, "directions must be a list of length equal to hidden_sizes - 1"
            for d in directions:
                assert d in [-1, 0, 1], "Each direction must be -1, 0 or 1"
        else:
            raise ValueError("directions must be an int or a list of ints")
        
        if isinstance(dropouts, float):
            dropout_vals = [dropouts] * (len(hidden_sizes) - 1)
        elif isinstance(dropouts, list):
            assert len(dropouts) == len(hidden_sizes) - 1, "dropout must be a float or a list of floats of length equal to hidden_sizes - 1"
            dropout_vals = dropouts
        else:
            raise ValueError("dropout must be a float or a list of floats")
        
        lstm_layers = []  
        #self.lstm_direction = [d for d in directions]
        dropouts_layers = [torch.nn.Dropout1d(p) for p in dropout_vals]
        self.dropout_layers = torch.nn.ModuleList(dropouts_layers)
        #prev_bidirectional = False
        for n in range(len(hidden_sizes) - 1):
            #if prev_bidirectional:
            #    in_sz = hidden_sizes[n] * 2
            #else:
            in_sz = hidden_sizes[n]
            out_sz = hidden_sizes[n+1]//2
            #direction = directions[n]
            lstm_layers.append(torch.nn.LSTM(input_size=in_sz, hidden_size=out_sz, batch_first=False, bidirectional=True))
            #prev_bidirectional = (direction == 0)
        self.lstm_layers = torch.nn.ModuleList(lstm_layers)
        #if prev_bidirectional:
        #    self.out_fc = torch.nn.Linear(hidden_sizes[-1]* 2, len(output_alphabet)+1)
        #else:
        self.out_fc = torch.nn.Linear((hidden_sizes[-1]//2) * 2, len(output_alphabet)+1)
        self.history = {'train_loss': [], 'valid_loss': {}, 'train_acc': [], 'valid_acc': {}, 'args': {}}
        print(f"Constructor {[layer.input_size for layer in self.lstm_layers]} -> {[layer.hidden_size for layer in self.lstm_layers]} and dropouts {dropout_vals}")

    @property
    def input_size(self) -> int:
        return len(self.input_mapper)
    @property
    def output_size(self) -> int:
        return len(self.output_mapper)

    def forward_bt(self, bt_x: torch.Tensor) -> torch.Tensor:
        btc_x = self.input_embedding(bt_x)
        tbc_x = btc_x.permute(1, 0, 2)  # Change to (batch, seq_len, embedding_dim)
        for layer, dropout in zip(self.lstm_layers, self.dropout_layers):
            tbc_x = torch.nn.functional.relu(tbc_x)
            tbc_x, _ = layer(tbc_x)
            tbc_x = dropout(tbc_x)
        tbc_x = torch.nn.functional.relu(tbc_x)
        btc_x = tbc_x.permute(1, 0, 2)  # Change back to (batch, seq_len, embedding_dim)
        btc_x = self.out_fc(btc_x)
        return btc_x
    
    def infer_str(self, src_str: str) -> str:
        src_tensor = self.input_mapper(src_str)
        with torch.no_grad():
            output = self.forward_bt(src_tensor)
        return self.output_mapper.decode(output)

    def is_compatible(self, other: Any) -> bool:
        if not isinstance(other, (DemapperLSTMCTC, Seq2SeqDs)):
            return False
        elif isinstance(other, Seq2SeqDs):
            return self.input_mapper.src_alphabet_str == other.input_mapper.src_alphabet_str and \
                   self.output_mapper.src_alphabet_str == other.output_mapper.src_alphabet_str
        elif isinstance(other, DemapperLSTMCTC):
            return self.input_mapper.src_alphabet_str == other.input_mapper.src_alphabet_str and \
                   self.output_mapper.src_alphabet_str == other.output_mapper.src_alphabet_str
        else:
            raise ValueError("Unsupported type for compatibility check")

    @property
    def hidden_sizes(self) -> List[int]:
        return [layer.input_size for layer in self.lstm_layers]

    @property
    def dropout_list(self) -> List[float]:
        return [layer.p for layer in self.dropout_layers]

    @property
    def epoch(self) -> int:
        return len(self.history['train_loss'])

    def save(self, path: str):
        dict_to_save = {
            'input_alphabet': self.input_mapper.src_alphabet_str,
            'output_alphabet': self.output_mapper.src_alphabet_str,
            'dropouts': self.dropout_list,
            'hidden_sizes': self.hidden_sizes,
            'state_dict': self.state_dict(),
            'history': self.history
        }
        torch.save(dict_to_save, path)
    
    @classmethod
    def __resume(cls, path: str) -> 'DemapperLSTMCTC':
        checkpoint = torch.load(path, map_location='cpu')
        input_alphabet = checkpoint['input_alphabet']
        output_alphabet = checkpoint['output_alphabet']
        dropouts = checkpoint['dropouts']
        hidden_sizes = checkpoint['hidden_sizes']
        model = cls(input_alphabet=input_alphabet, output_alphabet=output_alphabet, 
                    hidden_sizes=hidden_sizes, dropouts=dropouts)
        model.load_state_dict(checkpoint['state_dict'])
        model.history = checkpoint['history']
        return model
    
    @classmethod
    def resume(cls, path: str, input_alphabet_str: Optional[str] = None, output_alphabet_str: Optional[str] = None, hidden_sizes: List[int]=[128, 128, 128], 
                 dropouts: List[float] = [0.1, 0.1, 0.1]) -> 'DemapperLSTMCTC':
        try:
            res = cls.__resume(path)
        except FileNotFoundError:
            assert len(hidden_sizes) == len(dropouts), "hidden_sizes and dropouts must have the same length"
            assert input_alphabet_str is not None, "input_alphabet_str must be provided if path does not exist"
            assert output_alphabet_str is not None, "output_alphabet_str must be provided if path does not exist"            
            res = cls(input_alphabet=input_alphabet_str, output_alphabet=output_alphabet_str,
                      hidden_sizes=hidden_sizes, dropouts=dropouts)
        return res
    

    
    def get_one2one_train_objects(self, lr) -> Tuple[torch.optim.Optimizer, torch.nn.Module]:
        """Return the optimizer and criterion for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        return optimizer, criterion

    def validate_one2one_epoch(self, valid_ds: Seq2SeqDs, criterion: Optional[torch.nn.Module] = None, batch_size: int = 1, progress: bool = True) -> Tuple[float, float]:
        assert self.is_compatible(valid_ds), "The model is not compatible with the validation dataset."
        assert valid_ds.one2one_mapping, "The validation dataset must have one-to-one mapping."
        if self.history['valid_loss'].get(self.epoch) is not None and self.history['valid_acc'].get(self.epoch) is not None:
            return self.history['valid_loss'][self.epoch], self.history['valid_acc'][self.epoch]
        device = next(self.parameters()).device
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_lengths = 0
        confusion_matrix = torch.zeros(self.output_size, self.output_size, dtype=torch.int64)
        with torch.no_grad():
            for src_tensor_labels, tgt_tensor_labels in tqdm(valid_ds, disable=not progress, total=len(valid_ds)):
                src_tensor_labels = src_tensor_labels.to(device).unsqueeze(0)  # Add batch dimension
                tgt_tensor_labels = tgt_tensor_labels.unsqueeze(0).to(device)
                output = self.forward_bt(src_tensor_labels)
                loss = criterion(output.view(-1, output.size(-1)), tgt_tensor_labels.view(-1))
                total_loss += loss.item()
                _, predicted = torch.max(output, dim=-1)
                correct_np = (predicted == tgt_tensor_labels).cpu().numpy()
                error_idx = (~ correct_np).nonzero()
                total_correct += correct_np.sum()
                total_lengths += tgt_tensor_labels.numel()

        self.history['valid_loss'][self.epoch] = total_loss / len(valid_ds)
        self.history['valid_acc'][self.epoch] = total_correct / total_lengths if total_lengths > 0 else 0.0
        return self.history['valid_loss'][self.epoch], self.history['valid_acc'][self.epoch]


    def train_one2one_epoch(self, train_ds: Seq2SeqDs, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, batch_size: int = 1, pseudo_batch_size: int = 1, progress: bool = True) -> Tuple[float, float]:
        if batch_size > 1:
            raise NotImplementedError("Batch training is not implemented for DemapperLSTM. Use single instance training.")
        assert self.is_compatible(train_ds), "The model is not compatible with the training dataset."
        assert train_ds.one2one_mapping, "The training dataset must have one-to-one mapping."
        device = next(self.parameters()).device
        self.train()
        total_loss = 0.0
        total_correct = 0
        total_lengths = 0
        optimizer.zero_grad()
        try:
            desc = f"Training Epoch {self.epoch} Val acc: {list(self.history['valid_acc'].values())[-1]}"
        except IndexError:
            desc = f"Training Epoch {self.epoch} Val acc: N/A"
        for n, (src_tensor_labels, tgt_tensor_labels) in tqdm(enumerate(train_ds), total=len(train_ds), disable=not progress, desc=desc):
            src_tensor_labels = src_tensor_labels.to(device).unsqueeze(0)  # Add batch dimension
            tgt_tensor_labels = tgt_tensor_labels.unsqueeze(0).to(device)
            output = self.forward_bt(src_tensor_labels)
            loss = criterion(output.view(-1, output.size(-1)), tgt_tensor_labels.view(-1))
            loss.backward()
            if (n + 1) % pseudo_batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
            with torch.no_grad():
                _, predicted = torch.max(output, dim=-1)
                total_correct += (predicted == tgt_tensor_labels).sum().item()
                total_lengths += tgt_tensor_labels.numel()
        self.history['train_loss'].append(total_loss / len(train_ds))
        self.history['train_acc'].append(total_correct / total_lengths if total_lengths > 0 else 0.0)
        return self.history['train_loss'][-1], self.history['train_acc'][-1]

    def __repr__(self) -> str:
        return f"DemapperLSTM(input_alphabet={repr(self.input_mapper.src_alphabet_str)}, output_alphabet={repr(self.output_mapper.src_alphabet_str)}, " \
               f"hidden_sizes={repr(self.hidden_sizes)}, dropout={repr(self.dropout_list)})"
    
    def __str__(self):
        return self.__repr__() + f"Epoch: {self.epoch}\n" \
                f"\nTrain Accuracy: {self.history['train_acc'][-1] if self.history['train_acc'] else 'N/A'} \n" \
                f"Valid Accuracy: {self.history['valid_acc'].get(self.epoch, 'N/A')} \n" \
                f"Train Loss: {self.history['train_loss'][-1] if self.history['train_loss'] else 'N/A'} \n" \
                f"Valid Loss: {self.history['valid_loss'].get(self.epoch, 'N/A')} \n"


def main_train_one2one():
    import fargv
    from pathlib import Path
    from pylelemmatize.mapper_ds import Seq2SeqDs
    from pylelemmatize.fast_mapper import LemmatizerBMP
    import glob
    import tqdm
    from .charsets import allbmp_encoding_alphabet_strings
    import numpy as np
    import random
    

    p = {
        #"input_alphabet": allbmp_encoding_alphabet_strings["bmp_mufi"],
        "input_alphabet": allbmp_encoding_alphabet_strings["mes3a"],
        "output_alphabet": allbmp_encoding_alphabet_strings["ascii"],
        "hidden_sizes": "128,128,128",
        "corpus_files": set(glob.glob("./tmp/koeningsfelden/koenigsfelden_1308-1662_expanded/*0*txt")),
        "dropouts": "0.1,0.1,0.1",
        "batch_size": 1,
        "pseudo_batch_size": 1,
        "nb_epochs": 100,
        "num_workers": 8,
        "seed": 42,
        "output_model_path": "./tmp/model.pt",
        "train_test_split": 0.8,
        "lr": 0.001,
        "crop_seqlen": 0,  # Set to None to not crop the sequences
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "case_insensitive": True,  # If True, the input alphabet will be case insensitive
    }
    args, _ = fargv.fargv(p)
    args.hidden_sizes = [int(sz) for sz in args.hidden_sizes.split(',')]
    args.dropouts = [float(d) for d in args.dropouts.split(',')]
    
    if args.crop_seqlen <= 0:
        args.crop_seqlen = None

    random.seed(args.seed)
    
    corpus = '\n'.join([open(file, 'r', encoding='utf-8').read() for file in sorted(args.corpus_files)])
    corpus = corpus.split('\n')
    corpus = [line.strip() for line in corpus if line.strip()]
    corpus = [line for line in corpus if line]
    custom_map={'✳': '*', '*':'*'}
    #if args.case_insensitive:
    #    for c in args.input_alphabet:
    #        if c!= c.lower():
    #            custom_map[c] = c.lower()
    mapper = LemmatizerBMP.from_alphabet_mapping(src_alphabet_str=args.input_alphabet, dst_alphabet_str=args.output_alphabet, guess_unidecode=True, custom_map=custom_map)
    print(mapper("This is a test. with some ✳ characters. and their * replacements."))
    mapper = mapper.copy_removing_unused_inputs(''.join(corpus))  # Reduce the mapper to only the characters used in the corpus
    print(mapper("This is a test. with some ✳ characters. and their * replacements."))
    print(mapper, "Num outputs:", len(mapper.dst_alphabet_str), "Num inputs:", len(mapper.src_alphabet_str), "assignments:", len(mapper.mapping_dict))

    ds = Seq2SeqDs.create_selfsupervised_ds(corpus, mapper, mapped_is_input=True, crop_to_seqlen=args.crop_seqlen)
    train_ds, valid_ds = ds.split(args.train_test_split)
    wrong, total = 0, 0
    for src, tgt in ds:
        src_str = ds.input_mapper.intlabel_seq_to_str(src)
        tgt_str = ds.output_mapper.intlabel_seq_to_str(tgt)
        #print("STRINGS:", src_str, tgt_str)
        src = np.array(list(src_str), dtype=np.str_)
        tgt = np.array(list(tgt_str), dtype=np.str_)
        #print("NP STRINGS:", src, tgt)
        wrong += (src != tgt).sum()
        total += src.size
    print(f"Validation set mapper CER { wrong / total:.4f}")
    valid_ds.crop_seqlen = None  # Do not crop the validation dataset
    net = DemapperLSTMCTC.resume(args.output_model_path, 
                                input_alphabet_str=train_ds.input_mapper.src_alphabet_str, 
                                output_alphabet_str=train_ds.output_mapper.src_alphabet_str,
                                hidden_sizes=args.hidden_sizes, 
                                dropouts=args.dropouts)
    assert net.is_compatible(train_ds), "The model is not compatible with the training dataset."
    assert net.is_compatible(valid_ds), "The model is not compatible with the validation dataset."
    net = net.to(args.device)
    optimizer, criterion = net.get_one2one_train_objects(lr=args.lr)
    net.validate_one2one_epoch(valid_ds, criterion=criterion, batch_size=1)  # Validate before training
    print(net)
    net.save(args.output_model_path)
    while net.epoch < args.nb_epochs:
        print(f"Training epoch {net.epoch + 1}...")
        train_loss, train_acc = net.train_one2one_epoch(train_ds, criterion=criterion, optimizer=optimizer, batch_size=args.batch_size, pseudo_batch_size=args.pseudo_batch_size)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        valid_loss, valid_acc = net.validate_one2one_epoch(valid_ds, criterion=criterion, batch_size=args.batch_size)
        print(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}")
        net.save(args.output_model_path)