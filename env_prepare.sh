INSTALL_DIR="$HOME/miniconda"

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
bash miniconda.sh -b -p $INSTALL_DIR
rm miniconda.sh

export PATH="$INSTALL_DIR/bin:$PATH"
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

ENV_NAME="my_env"

conda create -y -n $ENV_NAME python=3.8

source activate $ENV_NAME