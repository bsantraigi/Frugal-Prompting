# Download DEB checkpoints
LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/EcQgyZ4J_i5MlIc8qSvQxZIBXFwIzrRjADW-_RLrLtCVfA?e=TJJIYd&download=1"
wget --content-disposition -c $LINK -O deb_model.zip
unzip -o deb_model.zip -d data/

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
