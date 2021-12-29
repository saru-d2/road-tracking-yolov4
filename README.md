# deepSORT YOLO + MOT analysis

Code for running experiments

## Made by [Tathagatho Roy](https://github.com/tathagatoroy) and [Saravanan Senthil](https://github.com/saru-d2)

follow instructions [here](./yolov4-deepsort/README.md) for how to run deepsort with a yolov4 model.

you can find our report [here](<link_to_report>)

## Instructions to download and use (modified) METEOR dataset

- download [Data_2](https://drive.google.com/drive/folders/1GCtt3bvyHY1s3PCEU8berxttu0j-e5MS?usp=sharing) and paste it in yolov4-deepsort
- ```python ./yolov4-deepsort/object_tracker.py --video ./Data_2/1/video_1.mp4 --output ./outputs/demo.avi --model yolov4```

## Instructions to get MOT metrics

- use config.ipynb to obtain MOT formatted data for all the annotations from the csv files generated from running deepsort.
- use the file structure outlined in [trail](./trial)
- run ```evaluate.py --dataPath <link-to-data>```

