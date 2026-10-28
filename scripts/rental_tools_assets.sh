set -euo pipefail
cd /workspace/agentic-image-pilot-small
mkdir -p models/tools third_party/archives
cp /model/HuggingFace/ShilongLiu/GroundingDINO/groundingdino_swint_ogc.pth models/tools/groundingdino_swint_ogc.pth
curl --fail --location --continue-at - --connect-timeout 15 --max-time 300 --retry 2 https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt -o models/tools/sam2.1_hiera_small.pt &
sam_pid=$!
curl --fail --location --continue-at - --connect-timeout 15 --max-time 300 --retry 2 https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth -o models/tools/dinov2_vitb14_pretrain.pth &
dino_pid=$!
for entry in 'IDEA-Research GroundingDINO 856dde20aee659246248e20734ef9ba5214f5e44' 'facebookresearch sam2 2b90b9f5ceec907a1c18123530e92e794ad901a4' 'facebookresearch dinov2 7764ea0f912e53c92e82eb78a2a1631e92725fc8'; do
    read -r owner name revision <<< "$entry"
    curl --fail --location --connect-timeout 15 --max-time 180 --retry 1 "https://gh-proxy.com/https://github.com/$owner/$name/archive/$revision.zip" -o "third_party/archives/$name-$revision.zip"
    /usr/local/miniconda3/envs/py312/bin/python -m zipfile -e "third_party/archives/$name-$revision.zip" third_party
    ln -sfn "$name-$revision" "third_party/$name"
done
wait "$sam_pid"
wait "$dino_pid"
ls -lh models/tools/*.p*
