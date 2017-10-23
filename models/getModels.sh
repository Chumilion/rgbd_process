# Downloading body pose model
OPENPOSE_URL="http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/"
COCO_MODEL="pose_iter_440000.caffemodel"
wget -c ${OPENPOSE_URL}${COCO_MODEL}
