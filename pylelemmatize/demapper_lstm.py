#from itertools import last
import sys
import time
import torch
from tqdm import tqdm

from pylelemmatize.mapper_ds import Seq2SeqDs
from .fast_mapper import LemmatizerBMP
from typing import Any, Union, List, Tuple, Dict, Literal, Optional
import numpy as np
from .util import print_err


class DemapperLSTM(torch.nn.Module):
    def __init__(self, input_mapper: Union[str, LemmatizerBMP], output_mapper: Union[str, LemmatizerBMP], hidden_sizes: List[int]=[128, 128, 128], 
                 dropouts: Union[List[float], float] = 0., directions: Union[Literal[-1, 0, 1], List[Literal[-1, 0, 1]]] = 0):
        super(DemapperLSTM, self).__init__()
        assert all([sz%2 == 0 for sz in hidden_sizes]), f"All hidden sizes must be even numbers. {hidden_sizes}"
        if isinstance(input_mapper, str):
            input_mapper = LemmatizerBMP(mapping_dict={c: c for c in input_mapper}, unknown_chr='�')
        if isinstance(output_mapper, str):
            output_mapper = LemmatizerBMP(mapping_dict={c: c for c in output_mapper}, unknown_chr='�')
        if isinstance(input_mapper, str):
            self.input_mapper = LemmatizerBMP(mapping_dict={c: c for c in input_mapper}, unknown_chr='�')
        else:
            self.input_mapper = input_mapper
        if isinstance(output_mapper, str):
            self.output_mapper = LemmatizerBMP(mapping_dict={c: c for c in output_mapper}, unknown_chr='�')
        else:
            self.output_mapper = output_mapper
        self.input_embedding = torch.nn.Embedding(len(input_mapper) + 1, hidden_sizes[0])
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
        dropouts_layers = [torch.nn.Dropout(p) for p in dropout_vals]
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
        self.out_fc = torch.nn.Linear((hidden_sizes[-1]//2) * 2, len(output_mapper)+1)
        self.history = {'train_loss': [], 'valid_loss': {}, 'train_acc': [], 'valid_acc': {}, 'args': {}, 'time_per_epoch': {0:time.time()}, 'best_weights': self.state_dict()}
        #print(f"Constructor {[layer.input_size for layer in self.lstm_layers]} -> {[layer.hidden_size for layer in self.lstm_layers]} and dropouts {dropout_vals}")

    @property
    def input_size(self) -> int:
        return len(self.input_mapper)
    @property
    def output_size(self) -> int:
        return len(self.output_mapper)

    def forward(self, bt_x: torch.Tensor) -> torch.Tensor:
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

    def infer_str(self, src_str: str, device: Optional[torch.cuda.device] = None, return_confidence: bool = False) -> str:
        if device is None:
            device = next(self.parameters()).device
        src_array = self.input_mapper.str_to_intlabel_seq(src_str)
        src_tensor = torch.tensor(src_array.astype(np.int64), dtype=torch.int64, device=device).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.forward(src_tensor)
        if return_confidence:
            output = torch.nn.functional.softmax(output, dim=-1)
            confidence = output.max(dim=-1).values.squeeze(0).cpu().numpy()  # Get confidence scores
        output = output.argmax(dim=-1).squeeze(0)  # Get the most probable output labels
        if return_confidence:
            return self.output_mapper.intlabel_seq_to_str(output.cpu().numpy()), confidence
        return self.output_mapper.intlabel_seq_to_str(output.cpu().numpy())

    def is_compatible(self, other: Any) -> bool:
        if not isinstance(other, (DemapperLSTM, Seq2SeqDs)):
            return False
        elif isinstance(other, Seq2SeqDs):
            return self.input_mapper.src_alphabet_str == other.input_mapper.src_alphabet_str and \
                   self.output_mapper.src_alphabet_str == other.output_mapper.src_alphabet_str
        elif isinstance(other, DemapperLSTM):
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

    def save(self, path: str, args: Optional[Any]= None):
        if args is not None:
            if 'args' not in self.history:
                self.history['args'] = {self.epoch: args}
            else:
                last_args = sorted(self.history['args'].items(), key=lambda x: x[0])
                last_args = last_args[-1][1] if len(last_args) > 0 else None
                if last_args != args:
                    self.history['args'][self.epoch] = args
        dict_to_save = {
            'input_alphabet': self.input_mapper,
            'output_alphabet': self.output_mapper,
            'dropouts': self.dropout_list,
            'hidden_sizes': self.hidden_sizes,
            'state_dict': self.state_dict(),
            'history': self.history
        }
        torch.save(dict_to_save, path)
    
    @classmethod
    def __resume(cls, path: str, resume_best_weights: bool) -> 'DemapperLSTM':
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        input_mapper = checkpoint['input_alphabet']
        output_mapper = checkpoint['output_alphabet']
        dropouts = checkpoint['dropouts']
        hidden_sizes = checkpoint['hidden_sizes']
        model = cls(input_mapper=input_mapper, output_mapper=output_mapper, 
                    hidden_sizes=hidden_sizes, dropouts=dropouts)
        if "best_weights" in checkpoint['history'] and resume_best_weights:
            model.load_state_dict(checkpoint['best_weights'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        model.history = checkpoint['history']
        return model
    
    @classmethod
    def resume(cls, path: str, input_alphabet_str: Optional[Union[str, LemmatizerBMP]] = None, output_alphabet_str: Optional[Union[str, LemmatizerBMP]] = None, hidden_sizes: List[int]=[128, 128, 128], 
                 dropouts: List[float] = [0.1, 0.1, 0.1], resume_best_weights: bool = False) -> 'DemapperLSTM':
        try:
            res = cls.__resume(path, resume_best_weights=resume_best_weights)
        except FileNotFoundError:
            assert len(hidden_sizes) == len(dropouts), "hidden_sizes and dropouts must have the same length"
            assert input_alphabet_str is not None, "input_alphabet_str must be provided if path does not exist"
            assert output_alphabet_str is not None, "output_alphabet_str must be provided if path does not exist"
            res = cls(input_mapper=input_alphabet_str, output_mapper=output_alphabet_str,
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
        with torch.no_grad():
            for src_tensor_labels, tgt_tensor_labels in tqdm(valid_ds, disable=not progress, total=len(valid_ds)):
                src_tensor_labels = src_tensor_labels.to(device).unsqueeze(0)  # Add batch dimension
                tgt_tensor_labels = tgt_tensor_labels.unsqueeze(0).to(device)
                output = self(src_tensor_labels)
                loss = criterion(output.view(-1, output.size(-1)), tgt_tensor_labels.view(-1))
                total_loss += loss.item()
                _, predicted = torch.max(output, dim=-1)
                correct_np = (predicted == tgt_tensor_labels).cpu().numpy()
                error_idx = (~ correct_np).nonzero()
                total_correct += correct_np.sum()
                total_lengths += tgt_tensor_labels.numel()
        self.history['valid_loss'][self.epoch] = total_loss / len(valid_ds)
        acc = total_correct / total_lengths if total_lengths > 0 else 0.0
        self.history['valid_acc'][self.epoch] = acc
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
            desc = f"Training Epoch {self.epoch} Val acc: {list(self.history['valid_acc'].values())[-1]:.6f}"
        except IndexError:
            desc = f"Training Epoch {self.epoch} Val acc: N/A"
        for n, (src_tensor_labels, tgt_tensor_labels) in tqdm(enumerate(train_ds), total=len(train_ds), disable=not progress, desc=desc):
            src_tensor_labels = src_tensor_labels.to(device).unsqueeze(0)  # Add batch dimension
            tgt_tensor_labels = tgt_tensor_labels.unsqueeze(0).to(device)
            output = self(src_tensor_labels)
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
        acc = total_correct / total_lengths if total_lengths > 0 else 0.0
        self.history['train_acc'].append(acc)
        if acc > max(self.history['train_acc'], default=-1.0):
            self.history["best_weights"] = self.state_dict()
        self.history['time_per_epoch'][self.epoch] = time.time()
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


def main_train_one2one(argv=sys.argv, **kwargs: Dict[str, Any]):
    import fargv
    from pathlib import Path
    from pylelemmatize.mapper_ds import Seq2SeqDs
    from pylelemmatize.fast_mapper import LemmatizerBMP
    import glob
    import tqdm
    #from .charsets import allbmp_encoding_alphabet_strings
    import pylelemmatize
    import numpy as np
    import random
    

    p = {
        #"input_alphabet": allbmp_encoding_alphabet_strings["bmp_mufi"],
        "input_alphabet": pylelemmatize.charset.mes3a,
        "output_alphabet": pylelemmatize.charset.ascii,
        "hidden_sizes": "128,128,128",
        "corpus_files": set(glob.glob("./tmp/koeningsfelden/koenigsfelden_1308-1662_expanded/*0*txt")),
        "dropouts": "0.1,0.1,0.1",
        "batch_size": 1,
        "pseudo_batch_size": 1,
        "nb_epochs": 100,
        "num_workers": 8,
        "seed": 42,
        "output_model_path": "./tmp/models/model.pt",
        "train_test_split": 0.8,
        "max_trainset_sz" : -1,  # -1 means no limit
        "lr": 0.001,
        "crop_seqlen": 0,  # Set to None to not crop the sequences
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "case_insensitive": True,  # If True, the input alphabet will be case insensitive
        "debug_sample": 3,
        "resume_best_weights": False,
        "custom_map": "",
        "min_char_similarity": 0.25,  # Minimum character similarity to consider a mapping valid
    }
    assert all([k in p for k, v in kwargs.items()])
    for k, v in kwargs.items():
        if k not in p:
            raise ValueError(f"Unknown argument {k}. Available arguments: {list(p.keys())}")
        if v is not None:
            assert isinstance(v, type(p[k])), f"Argument {k} must be of type {type(p[k])}, but got {type(v)} instead."  
    #assert all([type(v) == type(p[k]) for k, v in kwargs.items() if v is not None]), "All arguments must be provided."
    p.update(kwargs)
    args, _ = fargv.fargv(p, argv=argv)
    print(f"Running on cuda")
    args.hidden_sizes = [int(sz) for sz in args.hidden_sizes.split(',')]
    args.dropouts = [float(d) for d in args.dropouts.split(',')]
    
    if args.crop_seqlen <= 0:
        args.crop_seqlen = None

    random.seed(args.seed)
    
    corpus = '\n'.join([open(file, 'r', encoding='utf-8').read() for file in sorted(args.corpus_files)])
    corpus = corpus.split('\n')
    corpus = [line.strip() for line in corpus if line.strip()]
    corpus = [line for line in corpus if line]
    print(f"Corpus loaded: {len(corpus)} lines, {sum(len(line) for line in corpus)} characters, {len(set(''.join(corpus)))} unique characters.")
    if args.input_alphabet.strip() == "":
        args.input_alphabet = ''.join(set(''.join(corpus)))
    if args.output_alphabet.strip() == "":
        args.output_alphabet = ''.join(set(''.join(corpus)))
    if args.custom_map == "":
        custom_map = {}
    else:
        custom_map = {k.strip(): v.strip() for k, v in (item.split(':') for item in args.custom_map.split(','))}
        if len(custom_map) == 0:
            custom_map = {}
    mapper = LemmatizerBMP.from_alphabet_mapping(src_alphabet_str=args.input_alphabet, dst_alphabet_str=args.output_alphabet, override_map=custom_map, min_similarity=args.min_char_similarity)
    #print(mapper("This is a test. with some ✳ characters. and their * replacements."))
    mapper = mapper.copy_removing_unused_inputs(''.join(corpus))  # Reduce the mapper to only the characters used in the corpus
    print(mapper("This is a test. with some ✳ characters. and their * replacements."))
    print(mapper, "Num outputs:", len(mapper.dst_alphabet_str), "Num inputs:", len(mapper.src_alphabet_str), "assignments:", len(mapper.mapping_dict))

    ds = Seq2SeqDs.create_selfsupervised_ds(corpus, mapper, mapped_is_input=True, crop_to_seqlen=args.crop_seqlen)
    
    print(f"Dataset loaded: Items {len(ds)}, CER {ds.compute_ds_CER():.4f}% , Input alphabet: {ds.input_mapper.src_alphabet_str}, Output alphabet: {ds.output_mapper.src_alphabet_str}")

    train_ds, valid_ds = ds.split(args.train_test_split)
    if args.max_trainset_sz > 0:
        initial_train_size = len(train_ds)
        train_ds.src_text_blocks = train_ds.src_text_blocks[:args.max_trainset_sz]
        train_ds.tgt_text_blocks = train_ds.tgt_text_blocks[:args.max_trainset_sz]
        print(f"Reduced training dataset size from {initial_train_size} to {len(train_ds)} items.")
    print(f"Training Dataset : Lines {len(train_ds)}, Characters {sum(len(line) for line in train_ds.src_text_blocks)}, CER {train_ds.compute_ds_CER():.4f}%")
    print(f"Validation Dataset : Lines {len(valid_ds)}, Characters {sum(len(line) for line in valid_ds.src_text_blocks)}, CER {valid_ds.compute_ds_CER():.4f}%")
    print(f"Indicative Validation Sample:\n{valid_ds.render_sample(0)}\n")
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
    net = DemapperLSTM.resume(args.output_model_path, 
                                input_alphabet_str=train_ds.input_mapper.src_alphabet_str, 
                                output_alphabet_str=train_ds.output_mapper.src_alphabet_str,
                                hidden_sizes=args.hidden_sizes, 
                                dropouts=args.dropouts, resume_best_weights=args.resume_best_weights)
    assert net.is_compatible(train_ds), "The model is not compatible with the training dataset."
    assert net.is_compatible(valid_ds), "The model is not compatible with the validation dataset."
    net = net.to(args.device)
    optimizer, criterion = net.get_one2one_train_objects(lr=args.lr)
    net.validate_one2one_epoch(valid_ds, criterion=criterion, batch_size=1)  # Validate before training
    print(net)
    net.save(args.output_model_path, args=args)  # Save the initial model state
    while net.epoch < args.nb_epochs:
        print(f"Training epoch {net.epoch + 1}...")
        train_loss, train_acc = net.train_one2one_epoch(train_ds, criterion=criterion, optimizer=optimizer, batch_size=args.batch_size, pseudo_batch_size=args.pseudo_batch_size)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        valid_loss, valid_acc = net.validate_one2one_epoch(valid_ds, criterion=criterion, batch_size=args.batch_size)
        print(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}")
        for n in range(args.debug_sample):
            in_str, out_str = valid_ds[n]
            in_str = valid_ds.input_mapper.intlabel_seq_to_str(in_str)
            out_str = valid_ds.output_mapper.intlabel_seq_to_str(out_str)
            pred_str, confidence = net.infer_str(in_str, device=args.device, return_confidence=True)
            correct = (np.array(list(pred_str)) == np.array(list(out_str)))
            print("IN >",in_str)
            print("GT >",out_str)
            print("OUT>", end='')
            print_err(pred_str, correct=correct, confidence=confidence)
            print("")
        net.save(args.output_model_path, args=args)  # Save the model after each epoch


def main_report_demapper():
    import fargv
    p = {
        "models": set(["./tmp/models/model.pt",])
    }
    args, _ = fargv.fargv(p)
    for model_path in args.models:
        model = DemapperLSTM.resume(model_path)
        nb_epochs = len(model.history['train_loss'])
        validation_epochs = sorted(model.history['valid_accuray'].keys())



def main_infer_one2one():
    import fargv
    from pathlib import Path
    from pylelemmatize.mapper_ds import Seq2SeqDs
    from pylelemmatize.fast_mapper import LemmatizerBMP
    import tqdm
    import sys

    p = {
        "model_path": "./tmp/models/model.pt",
        "s" : "",
        "input_file": "stdin",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "print_line_count": False,
        "print_line_inputs": False,
        "print_line_rawinputs": False,
        "output_file": "stdout",
        "resume_last_weights": False,
        "allow_overwrite": False,
        "append_output": False,
        "add_newline": False,
        "new_line_separator": False,
    }
    args, _ = fargv.fargv(p)
    net = DemapperLSTM.resume(args.model_path, resume_best_weights=(not args.resume_last_weights))
    net = net.to(args.device)
    net.eval()
    def get_lines():
        if args.s != "":
            assert args.input_file in ["stdin", ""], "input_file must be 'stdin' or '' (empty string) if input_str is provided"
            for line in args.s.split('\n'):
                yield line
        elif args.input_file == "stdin":
            for line in sys.stdin:
                yield line[:-1] if line.endswith('\n') else line  # Remove trailing newline if present
        else:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    yield line[:-1] if line.endswith('\n') else line  # Remove trailing newline if present

    if args.output_file == "stdout":
        out_fd = sys.stdout
    else:
        assert not Path(args.output_file).exists() or args.allow_overwrite or args.append_output, f"Output file {args.output_file} already exists. Use --allow-overwrite to overwrite it."
        if args.append_output:
            out_fd = open(args.output_file, 'a', encoding='utf-8')
        else:
            out_fd = open(args.output_file, 'w', encoding='utf-8')
    column_separator = '\n' if args.new_line_separator else '\t'
    with torch.no_grad():
        for n, raw_line in enumerate(get_lines()):
            line = net.input_mapper(raw_line)
            #print(f"Processing line {n+1}: {line}", flush=True, file=sys.stderr)
            if args.print_line_count:
                print(f"{n}{column_separator}", end='', file=out_fd)
            if args.print_line_rawinputs:                
                print(f"{raw_line}{column_separator}", end='', file=out_fd)
            if args.print_line_inputs:                
                print(f"{line}{column_separator}", end='', file=out_fd)
            if len(line) == 0:
                output = ""
            else:
                output = net.infer_str(line)
            print(output, flush=True, file=out_fd)
            if args.add_newline:
                print("", flush=True, file=out_fd)