
import sys
from typing import Any, Dict


def main_train_substitution_only_postcorrection(argv=sys.argv, **kwargs: Dict[str, Any]):
    """Train a substitution-only HTR postcorrection model using LSTM."""
    import torch 
    import fargv
    from pathlib import Path
    from pylelemmatize.mapper_ds import Seq2SeqDs
    from pylelemmatize.fast_mapper import LemmatizerBMP
    import glob
    import tqdm
    #from .charsets import allbmp_encoding_alphabet_strings
    
    from .demapper_lstm import DemapperLSTM
    from .util import print_err
    from .mapper_ds import Seq2SeqDs
    #import pylelemmatize
    import numpy as np
    import random

    p = {
        #"input_alphabet": allbmp_encoding_alphabet_strings["bmp_mufi"],
        "trainset_tsv": "./experiments/htr_errors/tsv/pred_gt_trainset_substitions_only.tsv",
        "trainset_inputs": "",
        "trainset_groundtruth": "",
        "hidden_sizes": "256,256,256",
        "dropouts": "0.2,0.2,0.2",
        "batch_size": 1,
        "pseudo_batch_size": 1,
        "nb_epochs": 1000,
        "num_workers": 8,
        "seed": 1337,
        "output_model_path": "./tmp/models/postcorrection_model.pt",
        "train_test_split": 0.8,
        "max_trainset_sz" : -1,  # -1 means no limit
        "lr": 0.001,
        "crop_seqlen": 0,  # Set to None to not crop the sequences
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "debug_sample": 3,
        "resume_best_weights": False,
        "reverse_input_gt": False,
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

    #ds = Seq2SeqDs.create_selfsupervised_ds(corpus, mapper, mapped_is_input=True, crop_to_seqlen=args.crop_seqlen)
    if args.trainset_inputs == args.trainset_groundtruth == args.trainset_tsv == "":
        inputs, targets = zip(*[line.split("\t") for line in sys.stdin.readlines()])
    elif args.trainset_tsv != "":
        assert args.trainset_inputs == "" and args.trainset_groundtruth == "", "If trainset_tsv is provided, trainset_inputs and trainset_groundtruth must be empty"
        inputs, targets = zip(*[line.split("\t") for line in open(args.trainset_tsv, "r").readlines()])
    elif args.trainset_inputs != "" and args.trainset_groundtruth != "":
        assert args.trainset_tsv == "", "If trainset_inputs and trainset_groundtruth are provided, trainset_tsv must be empty"
        inputs = open(args.trainset_inputs, "r").readlines()
        targets = open(args.trainset_groundtruth, "r").readlines()
        assert len(inputs) == len(targets), "trainset_inputs and trainset_groundtruth must have the same number of lines"
    else:
        raise ValueError("Either trainset_tsv or both trainset_inputs and trainset_groundtruth must be provided")

    inputs = [line.replace("\n","") for line in inputs]
    targets = [line.replace("\n","") for line in targets]
    if args.reverse_input_gt:
        inputs, targets = targets, inputs

    for input, target in zip(inputs, targets):
        if len(input) != len(target):
            raise ValueError(f" Mismatched line lengths found in inputs or targets.\nInput: {repr(input)}\nTarget: {repr(target)}")


    ds = Seq2SeqDs(text_blocks=(inputs, targets))
    
    print(f"Dataset loaded: Items {len(ds)}, CER {ds.compute_ds_CER():.4f}% , Input alphabet: {ds.input_mapper.src_alphabet_str}, Output alphabet: {ds.output_mapper.src_alphabet_str}")

    train_ds, valid_ds = ds.split(args.train_test_split)
    validation_baseline_cer = valid_ds.compute_ds_CER()
    print(f"Validation set baseline CER (no model): {validation_baseline_cer:.4f}%")
    if args.max_trainset_sz > 0:
        initial_train_size = len(train_ds)
        train_ds.src_text_blocks = train_ds.src_text_blocks[:args.max_trainset_sz]
        train_ds.tgt_text_blocks = train_ds.tgt_text_blocks[:args.max_trainset_sz]
        print(f"Reduced training dataset size from {initial_train_size} to {len(train_ds)} items.")
    print(f"Training Dataset : Lines {len(train_ds)}, Characters {sum(len(line) for line in train_ds.src_text_blocks)}, CER {train_ds.compute_ds_CER():.4f}%")
    print(f"Validation Dataset : Lines {len(valid_ds)}, Characters {sum(len(line) for line in valid_ds.src_text_blocks)}, CER {valid_ds.compute_ds_CER():.4f}%")
    print(f"Indicative Validation Sample:\n{valid_ds.render_sample(0)}\n")

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
        print(f"Valid Loss: {valid_loss:.4f}, Valid CER: {100*(1-valid_acc):.4f}%, baseline CER (no model): {100*validation_baseline_cer:.4f}%")
        for n in range(args.debug_sample):
            in_str, out_str = valid_ds[n]
            in_str = valid_ds.input_mapper.intlabel_seq_to_str(in_str)
            out_str = valid_ds.output_mapper.intlabel_seq_to_str(out_str)
            pred_str, confidence = net.infer_str(in_str, device=args.device, return_confidence=True)
            correct = (np.array(list(pred_str)) == np.array(list(out_str)))
            print("IN >",in_str)
            print("GT >",out_str)
            print("OUT> ", end='')
            print_err(pred_str, correct=correct, confidence=confidence)
            print("")
        net.save(args.output_model_path, args=args)  # Save the model after each epoch


def main_postcorrection_infer():
    import torch
    import fargv
    from pathlib import Path
    from pylelemmatize.demapper_lstm import DemapperLSTM
    from pylelemmatize.util import print_err
    import sys
    import tqdm

    p = {
        "input_textlines": "",
        "output_textlines": "",
        "allow_overwrite": False,
        "correct_only_column": -1,
        "model_path": "./tmp/models/postcorrection_model.pt",
        "max_trainset_sz" : -1,  # -1 means no limit
        "crop_seqlen": 0,  # Set to None to not crop the sequences
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "verbose": False,
    }
    args, _ = fargv.fargv(p)
    if args.input_textlines == "":
        input_fd = sys.stdin
    else:
        input_fd = open( args.input_textlines, "r")

    if args.output_textlines == "":
        output_fd = sys.stdout
    else:
        if not Path(args.output_textlines).exists() or args.allow_overwrite:
            output_fd = open( args.output_textlines, "w")

    net = DemapperLSTM.resume(args.model_path)
    net = net.to(args.device)

    if args.verbose:
        progress = tqdm.tqdm(total=sum(1 for _ in input_fd.readlines()), desc="Processing lines")
        input_fd.seek(0)

    for input_line in input_fd.readlines():
        if input_line[-1] == "\n":
            input_line = input_line[:-1]
        if args.correct_only_column >= 0:
            line_pieces = input_line.split("\t")
            input_line = line_pieces[args.correct_only_column]
        #input_line = input_line
        if len(input_line.strip()) > 0:
            corrected_str, confidence = net.infer_str(input_line, device=args.device, return_confidence=True)
        else:
            corrected_str = input_line
        if args.correct_only_column >= 0:
            line_pieces[args.correct_only_column] = corrected_str
            corrected_str = "\t".join(line_pieces)
        print(corrected_str, file=output_fd)
        if args.verbose:
            progress.update(1)
    if args.verbose:
        progress.close()
    output_fd.flush()


def main_create_postcorrection_tsv():
    """creates a TSV where on substitutions are considered erros to train delemmatiser from arbitrary prediction-target pairs
    """
    import fargv
    import sys
    from pathlib import Path
    from pylelemmatize.substitution_augmenter import CharConfusionMatrix
    import tqdm
    import time
    p={
        "ocr_prediction_target_tsv":"",
        "substitution_only_tsv":"",
        "allow_overwrite": False,
        "min_line_length": 50,
        "max_edit_distance_tolerated": .2,
        "verbose": False,
    }
    args, _ = fargv.fargv(p)
    if args.ocr_prediction_target_tsv == "":
        input_fd = sys.stdin
    else:
        input_fd = open( args.ocr_prediction_target_tsv, "r")

    if args.substitution_only_tsv == "":
        output_fd = sys.stdout
    else:
        if not Path(args.substitution_only_tsv).exists() or args.allow_overwrite:
            output_fd = open( args.substitution_only_tsv, "w")
        else:
            raise IOError(f" Could not write to {args.substitution_only_tsv}")
    length_rejected = 0
    all_accepted = []
    for input_line in input_fd.readlines():
        input_line = input_line.strip().split("\t")
        if len(input_line) == 2 and len(input_line[1]) >= args.min_line_length:
            all_accepted.append(input_line)
        else:
            length_rejected+=1
    alphabet_str = "".join(sorted(set("".join([f"{p}{g}" for p, g in all_accepted]))))
    if args.verbose:
        print(f"Kept {len(all_accepted)} lines for processing, rejected {length_rejected} lines based on min_line_length {args.min_line_length}.", file=sys.stderr)
        print(f"Observed alphabet: {repr(alphabet_str)}", file=sys.stderr)
    
    cm = CharConfusionMatrix(alphabet=alphabet_str)
    t = time.time()
    ed_rejected = 0
    kept = 0
    if args.verbose:
        progress = tqdm.tqdm(total=len(all_accepted), desc="Processing lines")
    for pred, gt in all_accepted:
        no_sub, ed = cm.ingest_textline_observation(pred, gt)

        if len(no_sub) != len(gt):
            raise ValueError(f" Length mismatch after substitution-only processing\nPred: {pred}\nGT: {gt}\nNoSub: {no_sub}")
        if (ed / len(gt)) > args.max_edit_distance_tolerated:
            ed_rejected += 1
        else:
            print(f"{no_sub}\t{gt}", file=output_fd)
            kept += 1
        if args.verbose:
            progress.update(1)
    if args.verbose:
        progress.close()
    output_fd.flush()
    if args.verbose:
        print(f"Processed {ed_rejected + kept} lines, kept {kept}, rejected {ed_rejected} lines for edit distance > {args.max_edit_distance_tolerated} in {time.time() - t:.2f} sec.", file=sys.stderr)
