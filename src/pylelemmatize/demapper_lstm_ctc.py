import torch
from torch import Tensor
from typing import List, Union, Literal, Optional, Dict, Tuple, Any
import numpy as np
import time
import sys
from pathlib import Path
from tqdm import tqdm
from .many_to_more import ManyToMoreCollator, ManyToMoreDS, cer
from .demapper_lstm import DemapperLSTM
from .fast_mapper import LemmatizerBMP


class ManyToMoreCollatorCTC(ManyToMoreCollator):
    def __init__(self, ctc_epsilon_label:int =0, max_unalignment: int = -1, prefer_replication: bool = False):
        self.max_unalignment = max_unalignment
        self.ctc_epsilon_label = ctc_epsilon_label
        self.prefer_replication = prefer_replication

    def run_srcs_and_tgts(self, batch: List[Tuple[ Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        assert len(batch) == 1
        srcs, tgts = zip(*batch)
        assert len(srcs) == 1 and len(tgts) == 1
        srcs, tgts = srcs[0], tgts[0]
        if self.max_unalignment < 0:
            max_unalignment = max([len(tgt) for tgt in tgts])
        else:
            max_unalignment = self.max_unalignment
        src_tensor = torch.zeros((len(srcs) * (max_unalignment + 1)), dtype=torch.long) + self.ctc_epsilon_label
        tgts_tensor = torch.cat([tgt.long().view(-1) for tgt in tgts], dim=0)
        for i, src in enumerate(srcs.view(-1).tolist()):
            idx_start = i * (max_unalignment + 1)
            if self.prefer_replication:
                idx_end = idx_start + max_unalignment # an epsilon will be added at the end
            else:
                idx_end = idx_start + 1
            src_tensor[idx_start: idx_end] = src
        return src_tensor.unsqueeze(0), tgts_tensor.unsqueeze(0)

    def run_srcs(self, batch: List[Tensor]) -> Tensor:
        assert len(batch) == 1 and isinstance(batch[0], Tensor)
        srcs = batch[0]
        if self.max_unalignment < 0:
            max_unalignment = 1
        else:
            max_unalignment = self.max_unalignment
        src_tensor = torch.zeros((len(srcs) * (max_unalignment + 1)), dtype=torch.long) + self.ctc_epsilon_label
        for i, src in enumerate(srcs.view(-1).tolist()):
            idx_start = i * (max_unalignment + 1)
            if self.prefer_replication:
                idx_end = idx_start + max_unalignment # an epsilon will be added at the end
            else:
                idx_end = idx_start + 1
            src_tensor[idx_start: idx_end] = src 
        return src_tensor



class DemapperLSTMCTC(DemapperLSTM):
    def __init__(self, input_mapper: Union[str, LemmatizerBMP], output_mapper: Union[str, LemmatizerBMP],
                 hidden_sizes: List[int]=[128, 128, 128], 
                 dropouts: Union[List[float], float] = 0.,
                 directions: Union[Literal[-1, 0, 1], List[Literal[-1, 0, 1]]] = 0,
                 output_to_input_mapping: Optional[Dict[str, str]] = None, 
                 ctc_epsilon_label: int = 0,
                 max_unalignment: int = 10,
                 prefer_replication: bool = False,
                 ):
        super(DemapperLSTMCTC, self).__init__(input_mapper=input_mapper,
                                              output_mapper=output_mapper,
                                              hidden_sizes=hidden_sizes,
                                              dropouts=dropouts,
                                              directions=directions,
                                              output_to_input_mapping=output_to_input_mapping,
                                              )
        self.collator = ManyToMoreCollatorCTC(ctc_epsilon_label=ctc_epsilon_label, 
                                              max_unalignment=max_unalignment, 
                                              prefer_replication=prefer_replication)

    def forward(self, bt_x: Tensor) -> Tensor:
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

    def ctc_decode(self, btc_y: Tensor) -> Tuple[List[str], List[Tensor], List[Tensor]]:
        if btc_y.ndim == 3:
            bt_conf, bt_y = btc_y.max(dim=2)
        elif btc_y.ndim == 2:
            bt_y = btc_y
            bt_conf = torch.ones(bt_y.size(), dtype= torch.float())
        else:
            raise ValueError
        res_sequences = []
        res_confidences = []
        res_texts = []
        for batch_n in range(bt_y.size(0)):
            confidence = bt_conf[batch_n]
            most_probable = bt_y[batch_n]
            #confidence, most_probable = btc_y[batch_n,:-1].max(dim=1)
            padded_most_probable = torch.empty(bt_y.size(1)+1, dtype=torch.long)
            padded_most_probable[-1] = self.ctc_epsilon_label
            padded_most_probable[:-1] = most_probable
            non_duplicate = padded_most_probable[1:] != padded_most_probable[:-1]
            non_epsilon = padded_most_probable[:-1] != self.ctc_epsilon_label
            #print(most_probable)
            most_probable = most_probable[non_duplicate & non_epsilon]
            #print(most_probable)
            res_sequences.append(most_probable)
            res_confidences.append(confidence[non_duplicate & non_epsilon])
            res_texts.append(self.output_mapper.intlabel_seq_to_str(most_probable.cpu().numpy().astype(np.int16)))
        return res_texts, res_sequences, res_confidences

    def infer_str(self, src_str: str, device: Optional[torch.cuda.device] = None, return_confidence: bool = False) -> Union[str, Tuple[str, Tensor]]:
        if device is None:
            device = next(self.parameters()).device
        src_array = self.input_mapper.str_to_intlabel_seq(src_str)
        src_tensor = Tensor(src_array.astype(np.int64), dtype=torch.int64, device=device).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output_btc = self.forward(src_tensor)
        textlines, _, confidences = self.ctc_decode(output_btc)
        if return_confidence:
            return textlines[0], confidences[0]
        else:
            return textlines[0]

    def is_compatible(self, other: Any) -> bool:
        if not isinstance(other, (DemapperLSTMCTC, ManyToMoreDS)):
            return False
        elif isinstance(other, ManyToMoreDS):
            return self.input_mapper.src_alphabet_str == other.input_mapper.src_alphabet_str and \
                   self.output_mapper.src_alphabet_str == other.output_mapper.src_alphabet_str
        elif isinstance(other, DemapperLSTMCTC):
            return self.input_mapper.src_alphabet_str == other.input_mapper.src_alphabet_str and \
                   self.output_mapper.src_alphabet_str == other.output_mapper.src_alphabet_str
        else:
            raise ValueError("Unsupported type for compatibility check")

    @property
    def ctc_epsilon_label(self) -> int:
        return self.collator.ctc_epsilon_label
    
    @property
    def max_unalignment(self) -> int:
        return self.collator.max_unalignment
    
    @property
    def prefer_replication(self) -> bool:
        return self.collator.prefer_replication

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
            'history': self.history,
            'ctc_epsilon_label': self.ctc_epsilon_label,
            'max_unalignment': self.max_unalignment,
            'prefer_replication': self.prefer_replication
        }
        torch.save(dict_to_save, path)
    
    @classmethod
    def __resume(cls, path: str, resume_best_weights: bool) -> 'DemapperLSTMCTC':
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        input_mapper = checkpoint['input_alphabet']
        output_mapper = checkpoint['output_alphabet']
        dropouts = checkpoint['dropouts']
        hidden_sizes = checkpoint['hidden_sizes']
        ctc_epsilon_label = checkpoint['ctc_epsilon_label']
        max_unalignment = checkpoint['max_unalignment']
        prefer_replication = checkpoint['prefer_replication']
        model = cls(input_mapper=input_mapper, output_mapper=output_mapper, 
                    hidden_sizes=hidden_sizes, dropouts=dropouts,
                    ctc_epsilon_label=ctc_epsilon_label, max_unalignment=max_unalignment,
                    prefer_replication=prefer_replication)
        if "best_weights" in checkpoint['history'] and resume_best_weights:
            model.load_state_dict(checkpoint['history']['best_weights'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        model.history = checkpoint['history']
        model.output_to_input_mapping = model.history.get('output_to_input_mapping', {})
        return model
    
    @classmethod
    def resume(cls, path: str, input_alphabet_str: Optional[Union[str, LemmatizerBMP]] = None, output_alphabet_str: Optional[Union[str, LemmatizerBMP]] = None, hidden_sizes: List[int]=[128, 128, 128], 
                 dropouts: List[float] = [0.1, 0.1, 0.1], resume_best_weights: bool = False,
                 ctc_epsilon_label: int = 0, max_unalignment: int = 10,
                 prefer_replication: bool = False) -> 'DemapperLSTMCTC':
        if Path(path).is_file():
            res = cls.__resume(path, resume_best_weights=resume_best_weights)
        else:
            assert len(hidden_sizes) == len(dropouts), "hidden_sizes and dropouts must have the same length"
            assert input_alphabet_str is not None, "input_alphabet_str must be provided if path does not exist"
            assert output_alphabet_str is not None, "output_alphabet_str must be provided if path does not exist"
            res = cls(input_mapper=input_alphabet_str, output_mapper=output_alphabet_str,
                      hidden_sizes=hidden_sizes, dropouts=dropouts, ctc_epsilon_label=ctc_epsilon_label, 
                      max_unalignment=max_unalignment, prefer_replication=prefer_replication)
        return res
    
    def get_one2one_train_objects(self, lr) -> Tuple[torch.optim.Optimizer, torch.nn.Module]:
        """Return the optimizer and criterion for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CTCLoss(blank=self.ctc_epsilon_label)
        return optimizer, criterion

    def validate_one2one_epoch(self, valid_ds: ManyToMoreDS, criterion: Optional[torch.nn.Module] = None, batch_size: int = 1, progress: bool = True) -> Tuple[float, float]:
        if batch_size > 1:
            raise NotImplementedError()
        assert self.is_compatible(valid_ds), "The model is not compatible with the validation dataset."
        if self.history['valid_loss'].get(self.epoch) is not None and self.history['valid_acc'].get(self.epoch) is not None:
            return self.history['valid_loss'][self.epoch], self.history['valid_acc'][self.epoch]
        device = next(self.parameters()).device
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_lengths = 0
        valid_dl = torch.utils.data.DataLoader(dataset = valid_ds, batch_size=batch_size, shuffle=False, collate_fn=self.collator)
        with torch.no_grad():
            for src_tensor_labels, tgt_tensor_labels in tqdm(valid_dl, disable=not progress, total=len(valid_ds)):
                src_tensor_labels = src_tensor_labels.to(device)
                tgt_tensor_labels = tgt_tensor_labels.to(device)

                # TODO (anguelos) make this work with batch > 1
                assert src_tensor_labels.size(0) == src_tensor_labels.size(0) == 1, "Batch size greater than 1 is not supported yet."
                src_lengths = torch.tensor([[src_tensor_labels.size(1)]], dtype=torch.long)
                tgt_lengths = torch.tensor([[tgt_tensor_labels.size(1)]], dtype=torch.long)

                sparse_output = self(src_tensor_labels)

                loss = criterion(sparse_output.log_softmax(2)[0, :], tgt_tensor_labels[0, :],src_lengths.view(-1),tgt_lengths.view(-1))
                total_loss += loss.item()
                predicted_texts, predicted_labels, confidence = self.ctc_decode(sparse_output)

                for pred_lab, tgt_lab in zip(predicted_labels, tgt_tensor_labels):
                    if isinstance(pred_lab, Tensor):
                        pred_lab = pred_lab.cpu().numpy()
                    if isinstance(tgt_lab, Tensor):
                        tgt_lab = tgt_lab.cpu().numpy()
                    if len(pred_lab) == 0 or len(tgt_lab) == 0:
                        continue
                    total_correct += max(len(pred_lab)-cer(pred_lab, tgt_lab, normalize=False), 0)
                    total_lengths += len(tgt_lab)

        self.history['valid_loss'][self.epoch] = total_loss / len(valid_ds)
        acc = total_correct / total_lengths if total_lengths > 0 else 0.0
        if acc > max(self.history['valid_acc'].values(), default=-1.0):
            self.history["best_weights"] = self.state_dict()
        else:
            pass
        self.history['valid_acc'][self.epoch] = acc
        return self.history['valid_loss'][self.epoch], self.history['valid_acc'][self.epoch]

    def train_one2one_epoch(self, train_ds: ManyToMoreDS, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, batch_size: int = 1, pseudo_batch_size: int = 1, progress: bool = True) -> Tuple[float, float]:
        if batch_size > 1:
            raise NotImplementedError("Batch training is not implemented for Many to Many. Use single instance training.")
        assert self.is_compatible(train_ds), "The model is not compatible with the training dataset."
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
        train_dl = torch.utils.data.DataLoader(dataset = train_ds, batch_size=batch_size, shuffle=False, collate_fn=self.collator)
        for n, (src_tensor_labels, tgt_tensor_labels) in tqdm(enumerate(train_dl), total=len(train_ds), disable=not progress, desc=desc):
            src_tensor_labels = src_tensor_labels.to(device)
            tgt_tensor_labels = tgt_tensor_labels.to(device)

            # TODO (anguelos) make this work with batch > 1
            assert src_tensor_labels.size(0) == src_tensor_labels.size(0) == 1, "Batch size greater than 1 is not supported yet."
            src_lengths = torch.tensor([[src_tensor_labels.size(1)]], dtype=torch.long)
            tgt_lengths = torch.tensor([[tgt_tensor_labels.size(1)]], dtype=torch.long)

            sparse_output = self(src_tensor_labels)
            loss = criterion(sparse_output.log_softmax(2)[0, :], tgt_tensor_labels[0, :],src_lengths.view(-1),tgt_lengths.view(-1))
            loss.backward()
            if (n + 1) % pseudo_batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()

            with torch.no_grad():
                predicted_texts, predicted_labels, confidence = self.ctc_decode(sparse_output)
                for pred_lab, tgt_lab in zip(predicted_labels, tgt_tensor_labels):
                    if isinstance(pred_lab, Tensor):
                        pred_lab = pred_lab.cpu().numpy()
                    if isinstance(tgt_lab, Tensor):
                        tgt_lab = tgt_lab.cpu().numpy()
                    if len(pred_lab) == 0 or len(tgt_lab) == 0:
                        continue
                    total_correct += max(len(pred_lab)-cer(pred_lab, tgt_lab, normalize=False), 0)
                    total_lengths += len(tgt_lab)

        self.history['train_loss'].append(total_loss / len(train_ds))
        acc = total_correct / total_lengths if total_lengths > 0 else 0.0
        self.history['train_acc'].append(acc)
        if acc > max(self.history['train_acc'], default=-1.0):
            self.history["best_weights"] = self.state_dict()
        self.history['time_per_epoch'][self.epoch] = time.time()
        return self.history['train_loss'][-1], self.history['train_acc'][-1]


def main_train_many_to_more_ctc(argv=sys.argv, **kwargs: Dict[str, Any]):  # pragma: no cover
    import fargv
    from pathlib import Path
    from pylelemmatize.mapper_ds import Seq2SeqDs
    from pylelemmatize.fast_mapper import LemmatizerBMP
    from pylelemmatize.demapper_lstm_ctc import DemapperLSTMCTC
    from pylelemmatize.util import load_textline_pairs
    import glob
    import tqdm
    #from .charsets import allbmp_encoding_alphabet_strings
    import pylelemmatize
    import numpy as np
    import random
    p = {
        "inputs": set(glob.glob("/home/anguelos/data/corpora/maria_pia/abreviated/B*.xml")),
        "outputs": set(glob.glob("/home/anguelos/data/corpora/maria_pia/unabreviated/B*.xml")),
        "input_alphabet": pylelemmatize.charsets.mes3a,
        "output_alphabet": pylelemmatize.charsets.ascii,
        "hidden_sizes": "128,128,128",
        "dataset_cache_path": "./tmp/many_to_more_ds.pt",
        "allow_start_insertions": False,
        "band": 70,
        "verbose": False,
        "dropouts": "0.1,0.1,0.1",
        "pseudo_batch_size": 1,
        "nb_epochs": 100,
        "num_workers": 8,
        "seed": 42,
        "output_model_path": "./tmp/models/many_to_more_ctc.pt",
        "train_test_split": 0.8,
        "max_trainset_sz" : -1,  # -1 means no limit
        "lr": 0.001,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "debug_sample": 3,
        "resume_best_weights": False,
        "max_unalignment": -1,
        "batch_size": 1,
        "prefer_replication": False,
    }
    args, _ = fargv.fargv(p)
    args.hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    args.dropouts = [float(x) for x in args.dropouts.split(",")]
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if Path(args.dataset_cache_path).is_file():
        print(f"Dataset {args.dataset_cache_path} already exists. Loading existing dataset.", file=sys.stderr)
        dataset = ManyToMoreDS.load(args.dataset_cache_path, allow_start_insertions=args.allow_start_insertions)
    else:
        print(f"Creating dataset at {args.dataset_cache_path}", file=sys.stderr)
        line_pairs = load_textline_pairs(sorted(args.inputs), sorted(args.outputs))
        dataset = ManyToMoreDS.create_from_aligned_textlines(line_pairs=line_pairs, verbose=args.verbose, band=args.band, allow_start_insertions=args.allow_start_insertions)
        dataset.save(args.dataset_cache_path)
    train_ds, valid_ds = dataset.split(args.train_test_split)
    net = DemapperLSTMCTC.resume(path=args.output_model_path, 
                            input_alphabet_str=train_ds.input_mapper.src_alphabet_str, 
                            output_alphabet_str=train_ds.output_mapper.src_alphabet_str,
                            hidden_sizes=args.hidden_sizes, 
                            dropouts=args.dropouts)
    net = net.to(args.device)
    optimizer, ctc_loss = net.get_one2one_train_objects(args.lr)
    net.validate_one2one_epoch(valid_ds=valid_ds, criterion=ctc_loss, batch_size= args.batch_size, progress=args.verbose)
    net.save(args.output_model_path)
    while net.epoch < args.nb_epochs:
        print(f"Training epoch {net.epoch + 1}...")
        train_loss, train_acc = net.train_one2one_epoch(train_ds, criterion=ctc_loss, optimizer=optimizer,
                                                        batch_size=args.batch_size, pseudo_batch_size=args.pseudo_batch_size)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        valid_loss, valid_acc = net.validate_one2one_epoch(valid_ds, criterion=ctc_loss, batch_size=args.batch_size)
        print(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}")
        net.save(args.output_model_path)
