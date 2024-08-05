



<img width="713" alt="framework" src="https://github.com/user-attachments/assets/d19457e1-9de9-46c3-8fea-25ab09f9713b">





git clone git@github.com:llm-attacks/llm-attacks.git

git clone git@github.com:Gary3410/TaPA.git

Configure the preceding environment

For llm-attack:
pip install -e .

For TaPA:
cd TaPA
pip install -r requirements.txt

If you have problems with the installation, you can follow these steps

cd TaPA

pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117

pip install sentencepiece

pip install tqdm

pip install numpy

pip install jsonargparse[signatures]

pip install bitsandbytes

pip install datasets

pip install zstandard

pip install lightning==2.1.0.dev0

pip install deepspeed


cd ..
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

cd ..
git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
cd Detic
pip install -r requirements.txt

#example
python3 tapa-1key-harmful.py
