import argparse
import json
import logging
from pathlib import Path
import yaml
import sentencepiece as spm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_tokenizer(texts, model_dir, prefix, vocab_size=1000, model_type="unigram", char_coverage=0.9995):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    train_file = model_dir / "spm_input.txt"
    with open(train_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
    spm.SentencePieceTrainer.train(
        input=str(train_file),
        model_prefix=str(model_dir / prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=char_coverage,
    )
    return model_dir / f"{prefix}.model"


def load_transcripts(manifest_path):
    texts = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                texts.append(item["wrd"])
            except Exception as e:
                logger.warning(f"Skipping malformed line in manifest: {e}")
    return texts


def main():
    parser = argparse.ArgumentParser(description="CTC fine-tuning with SpeechBrain")
    parser.add_argument("hparams_file", type=Path, help="Path to YAML hyperparameters")
    parser.add_argument("--data_folder", type=Path)
    parser.add_argument("--output_folder", type=Path)
    args = parser.parse_args()

    hparams = yaml.safe_load(open(args.hparams_file))
    if args.data_folder:
        hparams["data_folder"] = str(args.data_folder)
    if args.output_folder:
        hparams["output_folder"] = str(args.output_folder)
    hparams.setdefault("save_folder", str(Path(hparams["output_folder"]) / "save"))

    train_manifest = Path(hparams["data_folder"]) / hparams["train_manifest"]
    texts = load_transcripts(train_manifest)
    model_file = train_tokenizer(
        texts,
        hparams["tokenizer_model_dir"],
        hparams.get("tokenizer_model_prefix", "spm"),
        hparams.get("tokenizer_vocab_size", 1000),
        hparams.get("tokenizer_model_type", "unigram"),
        hparams.get("tokenizer_char_coverage", 0.9995),
    )
    logger.info(f"Tokenizer model saved to {model_file}")

    try:
        import speechbrain as sb
        import torch
        from speechbrain.dataio.dataset import DynamicItemDataset
        from speechbrain.dataio.dataio import read_audio
        from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
    except Exception as e:
        logger.error("SpeechBrain or PyTorch not available: %s", e)
        return

    tokenizer = spm.SentencePieceProcessor(model_file=str(model_file))
    vocab_size = len(tokenizer)

    @sb.utils.data_pipeline.takes("wav", "wrd")
    @sb.utils.data_pipeline.provides("sig", "tokens", "tokens_lens")
    def audio_pipeline(wav, wrd):
        sig = read_audio(wav)
        tokens = tokenizer.encode_as_ids(wrd)
        return sig, torch.tensor(tokens, dtype=torch.int64), torch.tensor(len(tokens))

    datasets = {}
    for split in ["train", "valid", "test"]:
        manifest_key = f"{split}_manifest"
        if manifest_key not in hparams:
            continue
        manifest_file = Path(hparams["data_folder"]) / hparams[manifest_key]
        if not manifest_file.exists():
            continue
        dataset = DynamicItemDataset.from_json(manifest_file)
        dataset.add_dynamic_item(audio_pipeline)
        dataset.set_output_keys(["id", "sig", "tokens", "tokens_lens"])
        datasets[split] = dataset

    ssl_model = HuggingFaceWav2Vec2(source=hparams["ssl_model_hub"], output_norm=True)
    ctc_lin = torch.nn.Linear(ssl_model.model.config.hidden_size, vocab_size)

    modules = {"ssl_model": ssl_model, "ctc_lin": ctc_lin}

    class SimpleCTCBrain(sb.Brain):
        def compute_forward(self, batch, stage):
            wavs, wav_lens = batch.sig
            feats = self.modules.ssl_model(wavs)
            logits = self.modules.ctc_lin(feats)
            p_out = torch.nn.functional.log_softmax(logits, dim=-1)
            return p_out, wav_lens

        def compute_objectives(self, predictions, batch, stage):
            p_out, wav_lens = predictions
            tokens, tokens_lens = batch.tokens, batch.tokens_lens
            loss = sb.nnet.losses.ctc_loss(p_out, tokens, wav_lens, tokens_lens)
            return loss

    brain = SimpleCTCBrain(modules=modules,
                           opt_class=torch.optim.Adam,
                           hparams=hparams,
                           run_opts={"device": sb.core.auto_device()},
                           checkpointer=sb.utils.checkpoints.Checkpointer(hparams["save_folder"]))

    brain.fit(brain.hparams.get("epoch_counter", sb.utils.epoch_loop.EpochCounter(limit=1)),
              datasets.get("train"),
              train_loader_kwargs=hparams.get("train_dataloader_opts", {}),
              valid_set=datasets.get("valid"),
              valid_loader_kwargs=hparams.get("valid_dataloader_opts", {}))

    if "test" in datasets:
        brain.evaluate(datasets["test"], test_loader_kwargs=hparams.get("test_dataloader_opts", {}))


if __name__ == "__main__":
    main()
