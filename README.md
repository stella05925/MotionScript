
## Capture pose landmarks and convert to motionscript
```bash
cd mediapipe
python mediapipe_to_motionscript.py --source 0 --output my_motion.pt
```

## Run motionscript to generate caption
```bash
cd src
python run_captioning.py --input my_motion.pt --output_dir ./captions_out
```