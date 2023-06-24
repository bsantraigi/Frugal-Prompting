# Download DEB checkpoints
# Download the trained checkpoint files from https://drive.google.com/drive/folders/1N-_oFl26eGQM413zSQZ36ZFLTXzcpXqj
# Extract the downloaded zip to data/deb_model

export DEB_BASE_DIR=data/deb_model/random_only
# mkdir -p $DEB_BASE_DIR
curl "https://huggingface.co/bert-base-uncased/raw/main/config.json" -o $DEB_BASE_DIR/config.json
transformers-cli convert \
  --model_type bert \
  --tf_checkpoint $DEB_BASE_DIR/DEB_model.ckpt-3214 \
  --config $DEB_BASE_DIR/config.json \
  --pytorch_dump_output $DEB_BASE_DIR/pytorch_model.bin


export DEB_BASE_DIR=data/deb_model/random_and_adversarial
# mkdir -p $DEB_BASE_DIR
curl "https://huggingface.co/bert-base-uncased/raw/main/config.json" -o $DEB_BASE_DIR/config.json
transformers-cli convert \
  --model_type bert \
  --tf_checkpoint $DEB_BASE_DIR/DEB_model.ckpt-102870 \
  --config $DEB_BASE_DIR/config.json \
  --pytorch_dump_output $DEB_BASE_DIR/pytorch_model.bin
