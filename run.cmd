python tools/train.py -c configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml -t  output/rtdetrv2_r101vd_6x_coco/best.pth --test-only
python tools\export_onnx.py -c configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml -r output\rtdetrv2_r101vd_6x_coco\best.pth --check
python predict.py --onnx-file=model.onnx --im-dir=homework_dataset/valid/images
python predict.py --onnx-file=model.onnx --im-dir=homework_dataset/test/images