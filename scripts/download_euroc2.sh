mkdir -p datasets/euroc
cd datasets/euroc
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip
unzip MH_01_easy.zip -d mh01
rm -rf MH_01_easy.zip


wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_03_medium/MH_03_medium.zip
unzip MH_03_medium.zip -d mh03
rm -rf MH_03_medium.zip


wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_04_difficult/MH_04_difficult.zip
unzip MH_04_difficult.zip -d mh04
rm -rf MH_04_difficult.zip

wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_05_difficult/MH_05_difficult.zip
unzip MH_05_difficult.zip -d mh05
rm -rf MH_05_difficult.zip